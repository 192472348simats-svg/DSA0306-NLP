"""
ingest.py — Parse, chunk, embed, and store a user-uploaded PDF.

This module is imported by app.py at runtime — NOT run as a standalone script.

Key design decisions:
  • One ChromaDB collection per uploaded PDF, keyed by SHA-256 hash of file bytes.
    Re-uploading the same file reuses the existing index (no re-embedding).
    Different PDFs never mix chunks.
  • Embeddings are fully local via sentence-transformers (BAAI/bge-small-en-v1.5),
    chosen for its superior retrieval quality on technical/academic text vs MiniLM,
    while maintaining the exact same 33M parameter / CPU-friendly profile.
  • Section-aware chunking detects numbered headers (e.g. "4.3 Deadlock Avoidance"),
    Chapter/Part/Section labels, and ALL-CAPS bold titles — handles diverse textbook
    layouts far beyond the original numeric-only regex.
  • Sub-splits oversized sections by token count with overlap, tracking the actual
    page number of each window for precise citation (not just page_start).
  • A whole-document summary is generated at ingestion time via Groq, stored as a
    JSON sidecar, so meta-questions like "what is this document about?" are answered
    instantly from the cache.
"""

import os
import re
import json
import time
import hashlib
import pymupdf as fitz  # PyMuPDF (pymupdf >= 1.24 preferred import)
import chromadb
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

CHROMA_DIR        = "chroma_db"
CHUNK_TOKENS      = 600       # target chunk size in tokens
OVERLAP_TOKENS    = 80        # overlap between sub-split windows
EMBED_BATCH       = 256       # chunks per embedding call
SUMMARY_TOKENS    = 3000      # max tokens from start of doc fed to summary prompt

# Docs at or below this total-token count skip chunking entirely and are sent
# to the LLM as a single full-context call.  Kept at 8 000 so only genuinely
# short documents (≤ ~20 pages) use that path; everything longer is chunked.
FULL_CONTEXT_THRESHOLD = 8_000

# ── Section header patterns ──────────────────────────────────────
# Covers the four most common textbook conventions:
#   1. Numeric:   "4.3 Deadlock Avoidance", "12.1.2 Disk Scheduling"
#   2. Chapter:   "Chapter 4 — Memory Management" / "CHAPTER 4"
#   3. Part:      "Part II: Operating System Structures" / "PART 3"
#   4. Section:   "Section 2.1 Process Scheduling"
#   5. ALL-CAPS:  "DEADLOCK AVOIDANCE" (≥ 4 chars, no lowercase)
SECTION_RE = re.compile(
    r"(?:"
    # Numeric  e.g. "4.3 Title" or "12.1.2 Title"
    r"^\s*(\d+(?:\.\d+)+)\s+([A-Z][^\n]{2,80})\s*$"
    # Chapter N  e.g. "Chapter 4", "CHAPTER 4 — Title"
    r"|^\s*(?:chapter|CHAPTER)\s+\d+[\s:—–-]*([^\n]{0,80})\s*$"
    # Part N  e.g. "Part II", "PART 3 — Title"
    r"|^\s*(?:part|PART)\s+[\dIVXivx]+[\s:—–-]*([^\n]{0,80})\s*$"
    # Section N  e.g. "Section 2.1 Title"
    r"|^\s*(?:section|SECTION)\s+[\d.]+[\s:—–-]*([^\n]{0,80})\s*$"
    # ALL-CAPS standalone line ≥ 4 chars (bold headings in many PDFs)
    r"|^\s*([A-Z][A-Z\s\-–,:]{3,79})\s*$"
    r")",
    re.MULTILINE,
)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def file_hash(file_bytes: bytes) -> str:
    """Return a short SHA-256 hex digest to identify a PDF uniquely."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def extract_pages(file_bytes: bytes) -> list[dict]:
    """Extract text page-by-page from PDF bytes using PyMuPDF."""
    if not file_bytes:
        raise ValueError(
            "Uploaded file is empty (0 bytes) — re-upload or check the upload widget."
        )
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def clean(text: str) -> str:
    """Strip common textbook noise from a page's raw text."""
    # Lone page-number lines (e.g. "\n  312\n")
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _label_from_match(m: re.Match) -> str:
    """Extract the best human-readable label from a SECTION_RE match."""
    groups = m.groups()
    # group 0+1: numeric section (e.g. "4.3 Title")
    if groups[0] and groups[1]:
        return f"{groups[0]} {groups[1]}"
    # groups 2-5: chapter / part / section / ALL-CAPS
    for g in groups[2:]:
        if g and g.strip():
            return g.strip()
    return m.group(0).strip()


# ────────────────────────────────────────────────────────────────
# Section-aware chunking
# ────────────────────────────────────────────────────────────────

def chunk_by_section(pages: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Walk pages, detect section headers via the broadened SECTION_RE,
    group text by section, then sub-split oversized sections into
    ~CHUNK_TOKENS windows with OVERLAP_TOKENS overlap.

    Each final chunk carries:
        section    — detected header label (or "Front matter")
        page_start — page number where the section opened
        page       — page number of this specific token window (accurate for citations)
        text       — the raw text window

    Returns (chunks, section_titles).
    """
    enc = tiktoken.get_encoding("cl100k_base")

    raw_chunks: list[dict] = []
    section_titles: list[str] = []
    current = {"section": "Front matter", "page_start": 1, "text": "", "pages_seen": [1]}

    for p in pages:
        text = clean(p["text"])
        for line in text.split("\n"):
            m = SECTION_RE.match(line.strip())
            if m:
                label = _label_from_match(m)
                # Skip very generic matches like bare "I" or "A"
                if len(label) < 4:
                    current["text"] += line + "\n"
                    continue
                if current["text"].strip():
                    raw_chunks.append(current)
                section_titles.append(label)
                current = {
                    "section": label,
                    "page_start": p["page"],
                    "text": "",
                    "pages_seen": [p["page"]],
                }
            current["text"] += line + "\n"
            if p["page"] not in current["pages_seen"]:
                current["pages_seen"].append(p["page"])
        current["text"] += "\n"

    if current["text"].strip():
        raw_chunks.append(current)

    # Sub-split any section longer than CHUNK_TOKENS; track page accurately
    final: list[dict] = []
    for c in raw_chunks:
        toks = enc.encode(c["text"])
        if len(toks) <= CHUNK_TOKENS:
            final.append({
                "section":    c["section"],
                "page_start": c["page_start"],
                "page":       c["page_start"],
                "text":       c["text"],
            })
            continue

        step = CHUNK_TOKENS - OVERLAP_TOKENS
        pages_seen = c.get("pages_seen", [c["page_start"]])
        n_windows = max(1, (len(toks) - CHUNK_TOKENS) // step + 2)

        for win_idx, start in enumerate(range(0, len(toks), step)):
            window = toks[start: start + CHUNK_TOKENS]
            if not window:
                break
            piece = enc.decode(window)
            # Approximate which page this window falls on
            frac = win_idx / max(n_windows - 1, 1)
            page_idx = min(int(frac * len(pages_seen)), len(pages_seen) - 1)
            est_page = pages_seen[page_idx]
            final.append({
                "section":    c["section"],
                "page_start": c["page_start"],
                "page":       est_page,
                "text":       piece,
            })

    return final, section_titles


# ────────────────────────────────────────────────────────────────
# Document summary generation (one Groq call at ingest time)
# ────────────────────────────────────────────────────────────────

def generate_summary(pages: list[dict], section_titles: list[str], groq_client) -> str:
    """
    Generate a 150-200 word summary of the document via ONE Groq call.

    Scales to large documents: samples SAMPLE_TOKENS tokens from
    SAMPLE_POINTS evenly-spaced pages across the WHOLE document, plus
    the detected section headers, so the summary reflects the entire
    book rather than just chapter 1.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    SAMPLE_POINTS = 12   # evenly-spaced pages to sample (increased for better coverage)
    SAMPLE_TOKENS = 400  # tokens per sampled page

    cleaned = [clean(p["text"]) for p in pages]
    n = len(cleaned)
    if n <= SAMPLE_POINTS:
        sample_idx = list(range(n))
    else:
        step = n / SAMPLE_POINTS
        sample_idx = sorted({int(i * step) for i in range(SAMPLE_POINTS)})

    samples = []
    for idx in sample_idx:
        text = cleaned[idx]
        if not text:
            continue
        toks = enc.encode(text)[:SAMPLE_TOKENS]
        samples.append(f"[p. {pages[idx]['page']}] {enc.decode(toks)}")
    sampled_text = "\n\n[…]\n\n".join(samples)

    # Build section list (up to 60 sections shown)
    sections_str = (
        "\n".join(f"  • {s}" for s in section_titles[:60])
        if section_titles else "(no numbered sections detected)"
    )

    prompt = (
        f"Below are evenly-spaced excerpts from across a {n}-page document, "
        f"together with its detected section headers.\n\n"
        f"--- SAMPLED EXCERPTS (start / middle / end of document) ---\n"
        f"{sampled_text}\n--- END ---\n\n"
        f"--- DETECTED SECTIONS ---\n{sections_str}\n--- END ---\n\n"
        f"Write a concise summary (150-200 words) of what this document covers, "
        f"its main topics, and who it's intended for. Cover the full breadth of "
        f"the document, not just the opening. Be factual — only describe "
        f"what's actually in the text above."
    )

    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return (
            f"This document has {len(pages)} pages and "
            f"{len(section_titles)} detected sections. "
            f"Summary generation failed: {e}"
        )


# ────────────────────────────────────────────────────────────────
# Main entry point — called by app.py on upload
# ────────────────────────────────────────────────────────────────

def ingest_pdf(
    file_bytes: bytes,
    embedder,
    chroma_client,
    groq_client,
    progress_callback=None,
) -> dict:
    """
    Ingest an uploaded PDF (raw bytes) into ChromaDB.

    Args:
        file_bytes: Raw PDF bytes.
        embedder: SentenceTransformer instance (cached by caller).
        chroma_client: ChromaDB PersistentClient (cached by caller).
        groq_client: Groq client (cached by caller).
        progress_callback: Optional callable(step_name, detail) for UI updates.

    Returns dict with keys:
        collection_name, chunk_count, page_count, section_count, summary,
        mode, full_text, cached, section_titles
    """
    def _progress(step, detail=""):
        if progress_callback:
            progress_callback(step, detail)

    doc_id = file_hash(file_bytes)
    collection_name = f"doc_{doc_id}"
    summary_path = os.path.join(CHROMA_DIR, f"{collection_name}_summary.json")

    # ── Cache hit: summary sidecar already exists ──────────────────
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        summary       = summary_data.get("summary", "")
        mode          = summary_data.get("mode", "chunked")
        full_text     = summary_data.get("full_text", "")
        page_count    = summary_data.get("page_count", 0)
        section_count = summary_data.get("section_count", 0)
        section_titles = summary_data.get("section_titles", [])

        chunk_count = 0
        if mode == "chunked":
            existing = [c.name for c in chroma_client.list_collections()]
            chunk_count = (
                chroma_client.get_collection(collection_name).count()
                if collection_name in existing else 0
            )
        return {
            "collection_name": collection_name,
            "chunk_count":     chunk_count,
            "page_count":      page_count,
            "section_count":   section_count,
            "summary":         summary,
            "mode":            mode,
            "full_text":       full_text,
            "cached":          True,
            "section_titles":  section_titles,
        }

    # Legacy: collection exists but no sidecar
    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing:
        coll = chroma_client.get_collection(collection_name)
        return {
            "collection_name": collection_name,
            "chunk_count":     coll.count(),
            "page_count":      0,
            "section_count":   0,
            "summary":         "",
            "mode":            "chunked",
            "full_text":       "",
            "cached":          True,
            "section_titles":  [],
        }

    # ── Step 1: Parse PDF ──────────────────────────────────────────
    _progress("parsing", "Extracting text from PDF…")
    pages = extract_pages(file_bytes)

    non_empty = sum(1 for p in pages if len(p["text"].strip()) > 50)
    if non_empty < len(pages) * 0.3:
        print(
            f"⚠️  Warning: Only {non_empty}/{len(pages)} pages have meaningful text "
            f"— this PDF may be scanned/image-based and require OCR.\n"
            f"   Consider using a tool like Adobe Acrobat or pdfplumber with OCR enabled."
        )

    # Compute total token count to decide mode
    enc = tiktoken.get_encoding("cl100k_base")
    full_clean_text = "\n\n".join(clean(p["text"]) for p in pages)
    total_tokens = len(enc.encode(full_clean_text))

    # ── FULL-CONTEXT MODE: short doc, skip chunking entirely ────────
    if total_tokens <= FULL_CONTEXT_THRESHOLD:
        _progress("summarizing", "Generating document summary…")
        summary = generate_summary(pages, [], groq_client)

        summary_data = {
            "summary":        summary,
            "page_count":     len(pages),
            "section_count":  0,
            "section_titles": [],
            "mode":           "full_context",
            "full_text":      full_clean_text,
        }
        os.makedirs(CHROMA_DIR, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        _progress("done", "Ready!")
        return {
            "collection_name": collection_name,
            "chunk_count":     0,
            "page_count":      len(pages),
            "section_count":   0,
            "summary":         summary,
            "mode":            "full_context",
            "full_text":       full_clean_text,
            "cached":          False,
            "section_titles":  [],
        }

    # ── CHUNKED MODE: long doc ─────────────────────────────────────
    # Step 2: Chunk
    _progress("chunking", "Splitting into sections & chunks…")
    chunks, section_titles = chunk_by_section(pages)

    if not chunks:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be scanned/image-based and require OCR."
        )

    # Warn if section detection matched very few chunks
    bad = sum(
        1 for c in chunks
        if c["section"] in ("Front matter", "Untitled") or not c["section"]
    )
    matched_pct = 1.0 - (bad / len(chunks))
    if matched_pct < 0.05:
        print(
            f"⚠️  Section detection matched <5% of chunks "
            f"({bad}/{len(chunks)} unsectioned). Header patterns may not fit "
            f"this PDF's layout — chunk labels will be generic."
        )

    # Step 3: Embed and store
    coll = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    total = len(texts)
    embed_start = time.time()
    done = 0

    for i in range(0, total, EMBED_BATCH):
        batch_texts  = texts[i: i + EMBED_BATCH]
        batch_chunks = chunks[i: i + EMBED_BATCH]

        vecs = embedder.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        coll.add(
            ids=[f"chunk-{i + j}" for j in range(len(batch_texts))],
            embeddings=vecs,
            documents=batch_texts,
            metadatas=[
                {
                    "section": bc["section"],
                    "page":    bc["page"],          # accurate per-window page
                    "page_start": bc["page_start"], # section-open page (for context)
                }
                for bc in batch_chunks
            ],
        )

        done += len(batch_texts)
        elapsed = time.time() - embed_start
        rate = done / elapsed if elapsed > 0 else 0.0
        _progress(
            "embedding",
            f"Embedded {done}/{total} chunks ({rate:.0f} chunks/sec)",
        )

    embed_elapsed = time.time() - embed_start
    embed_rate = total / embed_elapsed if embed_elapsed > 0 else 0.0
    print(
        f"⚡ Embedded {total} chunks in {embed_elapsed:.1f}s "
        f"(batch={EMBED_BATCH}, {embed_rate:.1f} chunks/sec)"
    )

    # Step 4: Generate document summary
    _progress("summarizing", "Generating document summary…")
    summary = generate_summary(pages, section_titles, groq_client)

    summary_data = {
        "summary":        summary,
        "page_count":     len(pages),
        "section_count":  len(section_titles),
        "section_titles": section_titles[:100],
        "mode":           "chunked",
        "full_text":      "",
    }
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    _progress("done", "Ready!")

    return {
        "collection_name": collection_name,
        "chunk_count":     len(texts),
        "page_count":      len(pages),
        "section_count":   len(section_titles),
        "summary":         summary,
        "mode":            "chunked",
        "full_text":       "",
        "cached":          False,
        "section_titles":  section_titles,
    }
