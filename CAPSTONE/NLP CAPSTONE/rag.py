"""
rag.py — Retrieval-Augmented Generation using Groq + local BGE embeddings.

Exports:
    retrieve(question, collection_name, embedder, chroma_client, k=None)
    answer(question, collection_name, embedder, chroma_client, groq_client,
           history=None, mode="chunked", full_text="", query_expand=True)
    answer_stream(...)   — streaming variant; yields str tokens
    is_meta_question(question) → bool
    answer_from_summary(question, summary, groq_client) → str
    answer_full_context(question, full_text, groq_client, history=None) → str

All expensive clients (embedder, chroma, groq) are injected by the caller
so app.py can cache them with @st.cache_resource — no module-level loads.

Architecture:
  1. Query Expansion  — llama-3.1-8b-instant rewrites the user question into
     2 alternate phrasings; retrieval runs over all variants and unions results.
  2. Semantic Retrieval — BGE embeddings + ChromaDB cosine search, TOP_K candidates.
  3. BM25 Re-ranking   — rank_bm25 keyword overlap re-scores candidates; top
     FINAL_K chunks are kept.  Pure Python, zero GPU, negligible CPU.
  4. LLM Generation    — llama-3.3-70b-versatile (streaming) with a friendly,
     grounded system prompt that cites page numbers.
"""

import re
from rank_bm25 import BM25Okapi

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

# Primary LLM — most capable model available on this Groq account
MODEL           = "openai/gpt-oss-120b"
# Faster model for cheap query expansion (< 200 tokens in/out)
EXPAND_MODEL    = "openai/gpt-oss-20b"

MIN_SIMILARITY  = 0.28    # cosine similarity floor (BGE has better separation, safe to lower)
TOP_K           = 10      # initial semantic candidates per query variant
TOP_K_LARGE     = 14      # candidates for large collections (> LARGE_DOC_CHUNKS)
FINAL_K         = 6       # chunks kept after BM25 re-ranking
LARGE_DOC_CHUNKS = 500    # chunk-count threshold for "large document"
TEMPERATURE     = 0.25    # slightly above 0.2 for a warmer, more natural tone
MAX_TOKENS      = 900     # allow slightly longer answers
HISTORY_TURNS   = 8       # conversation turns kept as context

# ────────────────────────────────────────────────────────────────
# System prompts
# ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a friendly and knowledgeable study buddy helping a student understand \
their textbook. You answer questions ONLY using the excerpts provided below, \
taken directly from their uploaded document.

TONE & STYLE:
• Be warm, clear, and encouraging — like a brilliant friend who happens to know \
the subject inside-out.
• Use plain language first, then precise terminology. Analogies are welcome \
when they make things click.
• Structure longer answers with bullet points or numbered steps; use **bold** \
for key terms.
• Keep the answer focused and exam-relevant unless the student explicitly asks \
for more depth.

ACCURACY RULES (non-negotiable):
1. Base every factual claim on the provided excerpts — nothing else.
   If the excerpts don't cover the question, say so honestly:
   "I couldn't find that in your textbook — try rephrasing, or it may be in \
a section not yet uploaded."
2. Cite the source like (Sec. 4.3, p. 312) or (p. 78) after every fact.
3. If excerpts say conflicting things, flag the discrepancy rather than picking \
one silently.
4. Never guess, speculate, or draw on outside knowledge."""

FULL_CONTEXT_SYSTEM_PROMPT = """\
You are a friendly and knowledgeable study buddy. The student has uploaded a \
short document and you have the FULL text of it below.

TONE & STYLE:
• Be warm, clear, and encouraging — like a brilliant friend who happens to know \
the subject inside-out.
• Use plain language first, then precise terminology. Analogies are welcome.
• Structure longer answers with bullet points or numbered steps; use **bold** \
for key terms.

ACCURACY RULES (non-negotiable):
1. The ENTIRE document is in front of you. If the answer is not in it, say:
   "That doesn't appear to be covered in your document."
   Never guess or use outside knowledge.
2. Cite page numbers where visible using [p. N] markers in the text.
3. Use conversation history to resolve follow-ups ("what are they?" → refers to \
previous turn)."""

# ────────────────────────────────────────────────────────────────
# Meta-question detection
# ────────────────────────────────────────────────────────────────

_META_PATTERNS = re.compile(
    r"(what\s+(is|does)\s+this\s+(pdf|document|book|textbook|file|paper)\s+(about|cover|contain))"
    r"|(summarize|summarise|summary|overview|table\s+of\s+contents|toc)"
    r"|(what\s+topics|what\s+are\s+the\s+(main|key)\s+(topics|chapters|sections|concepts))"
    r"|(give\s+me\s+(a|an)\s+(summary|overview|brief))"
    r"|(what\s+is\s+this\s+about)"
    r"|(describe\s+this\s+(document|pdf|book|textbook))"
    r"|(what\s+does\s+this\s+cover)"
    # Chapter/section meta queries
    r"|(what\s+(does|is\s+in)\s+(chapter|section|part)\s+\d)"
    r"|(list\s+(all\s+)?(topics|chapters|sections|concepts|parts))"
    r"|(what\s+chapters|which\s+chapters)",
    re.IGNORECASE,
)


def is_meta_question(question: str) -> bool:
    """
    Detect whether a question is asking about the document as a whole
    (overview / summary / TOC / chapter list) rather than a specific factual
    question.  Uses regex only — no LLM call.
    """
    q = question.strip().lower()
    if _META_PATTERNS.search(q):
        return True
    return False


def answer_from_summary(question: str, summary: str, groq_client) -> str:
    """Answer a meta-question using the pre-generated document summary."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly study assistant. The student is asking about "
                "a document they uploaded. Answer using ONLY the document summary "
                "below. Be warm, concise, and helpful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document summary:\n{summary}\n\n"
                f"Question: {question}"
            ),
        },
    ]
    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content


# ────────────────────────────────────────────────────────────────
# Query expansion — optional, uses fast llama-3.1-8b-instant
# ────────────────────────────────────────────────────────────────

def expand_query(question: str, groq_client) -> list[str]:
    """
    Rewrite the user's question into 2 alternate phrasings using the tiny
    llama-3.1-8b-instant model.  Returns [original, variant1, variant2].
    On any failure, silently falls back to [original].
    """
    try:
        prompt = (
            "You are a query-expansion assistant for a textbook search engine.\n"
            "Given the student's question, produce exactly 2 alternative phrasings "
            "that use different vocabulary but ask for the same information.\n"
            "Each phrasing on its own line. No numbering, no bullets, no explanation.\n\n"
            f"Question: {question}"
        )
        resp = groq_client.chat.completions.create(
            model=EXPAND_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=80,
        )
        lines = [
            l.strip()
            for l in resp.choices[0].message.content.strip().splitlines()
            if l.strip()
        ][:2]
        return [question] + lines
    except Exception:
        return [question]


# ────────────────────────────────────────────────────────────────
# BM25 re-ranking — pure Python, zero GPU
# ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace+lowercase tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_rerank(question: str, chunks: list[dict], top_n: int = FINAL_K) -> list[dict]:
    """
    Re-rank *chunks* by BM25 keyword relevance to *question* and return
    the top_n best chunks.  Falls back to the original order if BM25 fails
    (e.g. empty corpus).
    """
    if not chunks:
        return chunks
    try:
        corpus = [_tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(corpus)
        query_toks = _tokenize(question)
        scores = bm25.get_scores(query_toks)
        # Blend semantic score (from retrieval) with BM25 score
        max_bm25 = max(scores) if max(scores) > 0 else 1.0
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: 0.6 * x[0]["score"] + 0.4 * (x[1] / max_bm25),
            reverse=True,
        )
        return [c for c, _ in ranked[:top_n]]
    except Exception:
        return chunks[:top_n]


# ────────────────────────────────────────────────────────────────
# Retrieval — embed query locally, search ChromaDB, threshold-filter
# ────────────────────────────────────────────────────────────────

def retrieve(
    question: str,
    collection_name: str,
    embedder,
    chroma_client,
    k: int | None = None,
) -> list[dict]:
    """
    Embed the question and return the top-k chunks from the named ChromaDB
    collection that pass the similarity threshold.

    k is chosen dynamically: large collections (> LARGE_DOC_CHUNKS) get
    TOP_K_LARGE candidates; smaller collections use TOP_K.
    """
    coll = chroma_client.get_collection(collection_name)
    if k is None:
        k = TOP_K_LARGE if coll.count() > LARGE_DOC_CHUNKS else TOP_K
    q_vec = embedder.encode(
        [question], normalize_embeddings=True
    ).tolist()[0]

    res = coll.query(
        query_embeddings=[q_vec],
        n_results=min(k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        similarity = 1 - dist
        chunks.append({
            "text":    doc,
            "section": meta.get("section", ""),
            "page":    meta.get("page", meta.get("page_start", 0)),
            "score":   round(similarity, 3),
        })

    passing = [c for c in chunks if c["score"] >= MIN_SIMILARITY]
    if passing:
        return passing
    # Threshold fallback — return top-3 raw so SYSTEM_PROMPT handles grounding
    return chunks[:3]


def retrieve_multi(
    queries: list[str],
    collection_name: str,
    embedder,
    chroma_client,
) -> list[dict]:
    """
    Run retrieve() for each query variant and union the results by text,
    keeping the highest similarity score for duplicates.
    """
    seen: dict[str, dict] = {}
    for q in queries:
        for chunk in retrieve(q, collection_name, embedder, chroma_client):
            key = chunk["text"][:120]  # dedup key
            if key not in seen or chunk["score"] > seen[key]["score"]:
                seen[key] = chunk
    return list(seen.values())


# ────────────────────────────────────────────────────────────────
# Generation — build context from surviving chunks, call Groq
# ────────────────────────────────────────────────────────────────

def answer(
    question: str,
    collection_name: str,
    embedder,
    chroma_client,
    groq_client,
    history: list[dict] | None = None,
    mode: str = "chunked",
    full_text: str = "",
    query_expand: bool = True,
) -> tuple[str, list[dict]]:
    """
    Branch on the document's ingestion mode:
      • full_context: short doc — send ENTIRE text + recent history to Groq.
      • chunked: long doc — expand query, retrieve, BM25 re-rank, generate.

    Returns (answer_text, source_chunks).
    """
    if mode == "full_context" and full_text:
        return answer_full_context(question, full_text, groq_client, history), []

    # --- Chunked mode ---
    queries = expand_query(question, groq_client) if query_expand else [question]
    all_chunks = retrieve_multi(queries, collection_name, embedder, chroma_client)

    if not all_chunks:
        return (
            "Hmm, I couldn't find anything relevant to that in your textbook. "
            "Try rephrasing your question, or check if that topic is in a different "
            "chapter that hasn't been uploaded yet.",
            [],
        )

    # BM25 re-rank and select the best FINAL_K chunks
    chunks = bm25_rerank(question, all_chunks)

    # Build labelled context
    context_parts = [
        f"[{c['section']}, p. {c['page']}]\n{c['text']}"
        for c in chunks
    ]
    context = "\n\n---\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend((history or [])[-HISTORY_TURNS:])
    messages.append({
        "role": "user",
        "content": (
            f"Excerpts from your textbook:\n\n{context}\n\n"
            f"---\n\nStudent question: {question}"
        ),
    })

    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return resp.choices[0].message.content, chunks


def answer_stream(
    question: str,
    collection_name: str,
    embedder,
    chroma_client,
    groq_client,
    history: list[dict] | None = None,
    mode: str = "chunked",
    full_text: str = "",
    query_expand: bool = True,
):
    """
    Streaming variant of answer().

    Yields:
        ("chunks", list[dict])  — source chunks (first, before any text)
        ("token", str)          — each streamed token from the LLM
        ("done", "")            — signals completion

    The caller (app.py) iterates this generator and handles each event type.
    """
    if mode == "full_context" and full_text:
        yield ("chunks", [])
        messages = [{"role": "system", "content": FULL_CONTEXT_SYSTEM_PROMPT}]
        messages.extend((history or [])[-HISTORY_TURNS:])
        messages.append({
            "role": "user",
            "content": (
                f"FULL DOCUMENT TEXT:\n\n{full_text}\n\n"
                f"---\n\nQuestion: {question}"
            ),
        })
        stream = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield ("token", delta)
        yield ("done", "")
        return

    # --- Chunked mode ---
    queries = expand_query(question, groq_client) if query_expand else [question]
    all_chunks = retrieve_multi(queries, collection_name, embedder, chroma_client)

    if not all_chunks:
        yield ("chunks", [])
        yield ("token",
               "Hmm, I couldn't find anything relevant to that in your textbook. "
               "Try rephrasing your question, or check if that topic is in a different "
               "chapter that hasn't been uploaded yet.")
        yield ("done", "")
        return

    chunks = bm25_rerank(question, all_chunks)
    yield ("chunks", chunks)

    context_parts = [
        f"[{c['section']}, p. {c['page']}]\n{c['text']}"
        for c in chunks
    ]
    context = "\n\n---\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend((history or [])[-HISTORY_TURNS:])
    messages.append({
        "role": "user",
        "content": (
            f"Excerpts from your textbook:\n\n{context}\n\n"
            f"---\n\nStudent question: {question}"
        ),
    })

    stream = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield ("token", delta)
    yield ("done", "")


# ────────────────────────────────────────────────────────────────
# Full-context generation (short docs — entire text in one call)
# ────────────────────────────────────────────────────────────────

def answer_full_context(
    question: str,
    full_text: str,
    groq_client,
    history: list[dict] | None = None,
) -> str:
    """
    Answer using the ENTIRE document text + conversation history in a
    single Groq call.  No retrieval, no similarity filtering.
    """
    messages = [{"role": "system", "content": FULL_CONTEXT_SYSTEM_PROMPT}]
    messages.extend((history or [])[-HISTORY_TURNS:])
    messages.append({
        "role": "user",
        "content": (
            f"FULL DOCUMENT TEXT:\n\n{full_text}\n\n"
            f"---\n\nQuestion: {question}"
        ),
    })
    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content
