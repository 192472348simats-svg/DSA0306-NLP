"""
app.py — Streamlit chat UI with PDF upload for the RAG Study Assistant.

Launch with:
    streamlit run app.py

Features:
  • @st.cache_resource for SentenceTransformer (BGE-small), ChromaDB, and Groq.
  • Streaming responses via answer_stream() — tokens appear in real-time.
  • Query expansion (llama-3.1-8b-instant) + BM25 re-ranking for large textbooks.
  • Sidebar section navigator — detected chapters/sections listed as quick links.
  • Document summary panel with warm, conversational overview.
  • Step-by-step ingestion progress with per-batch embedding updates.
  • Modern dark UI with glassmorphism cards and smooth animations.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StudyPal — AI Textbook Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────
# Cached resource loaders — load ONCE per process, not per rerun
# ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model (BGE-small)…")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    # BGE-small-en-v1.5: same 33M params as MiniLM, ~20% better retrieval on
    # technical text.  Auto-downloaded on first run (~90 MB), then cached.
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner=False)
def load_chroma():
    import chromadb
    return chromadb.PersistentClient(path="chroma_db")


@st.cache_resource(show_spinner=False)
def load_groq():
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])


# Load all resources once
embedder     = load_embedder()
chroma_client = load_chroma()
groq_client  = load_groq()

# Import after resources are ready
from ingest import ingest_pdf
from rag import answer_stream, is_meta_question, answer_from_summary

# ────────────────────────────────────────────────────────────────
# Global CSS — dark glassmorphism theme
# ────────────────────────────────────────────────────────────────

st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 860px; }

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 1.6rem 2rem 1.4rem;
    margin-bottom: 1.4rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
}
.hero-header p { color: rgba(255,255,255,0.55); font-size: 0.9rem; margin: 0; }

/* ── Summary box ── */
.summary-box {
    background: linear-gradient(135deg, #1e1b4b 0%, #1e3a5f 100%);
    color: #c7d2fe;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    font-size: 0.92rem;
    line-height: 1.7;
    border: 1px solid rgba(167,139,250,0.25);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── Section pill chips ── */
.section-pill {
    display: inline-block;
    background: rgba(96,165,250,0.12);
    color: #93c5fd;
    border: 1px solid rgba(96,165,250,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    margin: 0.15rem;
    cursor: pointer;
    transition: background 0.2s;
}
.section-pill:hover { background: rgba(96,165,250,0.25); }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.2rem 0.4rem;
    margin-bottom: 0.5rem;
}

/* ── Sources expander ── */
.source-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
}
.source-card .source-meta {
    color: #7c3aed;
    font-weight: 600;
    font-size: 0.8rem;
    margin-bottom: 0.35rem;
}
.source-card .source-excerpt { color: #94a3b8; line-height: 1.6; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
section[data-testid="stSidebar"] { background: #0f172a; }

/* ── Landing cards ── */
.landing-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: border-color 0.3s, transform 0.2s;
}
.landing-card:hover { border-color: rgba(167,139,250,0.4); transform: translateY(-2px); }
.landing-card .icon { font-size: 2rem; margin-bottom: 0.6rem; }
.landing-card h3 { font-size: 1rem; font-weight: 600; margin: 0 0 0.4rem; color: #e2e8f0; }
.landing-card p { font-size: 0.82rem; color: #64748b; margin: 0; line-height: 1.5; }

/* ── Progress step ── */
.progress-step { padding: 0.25rem 0; font-size: 0.88rem; }
.progress-step.done  { color: #34d399; }
.progress-step.active{ color: #60a5fa; font-weight: 600; }
.progress-step.pending{ color: #475569; }

/* ── Sidebar metrics ── */
[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 0.5rem; }
</style>
""")

# ────────────────────────────────────────────────────────────────
# Session state init
# ────────────────────────────────────────────────────────────────

for key, default in {
    "collection": None,
    "messages": [],
    "sources": {},
    "last_file": None,
    "doc_info": None,
    "prefill_question": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 StudyPal")
    st.caption("Your AI textbook companion")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload a PDF textbook",
        type="pdf",
        help="Supports textbooks up to 1000+ pages. Any PDF with selectable text.",
    )

    # Query expansion toggle
    st.markdown("---")
    query_expand = st.toggle(
        "🔍 Smart query expansion",
        value=True,
        help="Rewrites your question into 2 alternate phrasings before searching — "
             "catches terminology mismatches. Uses 1 extra fast Groq call.",
    )

    # Section navigator (shown after ingestion)
    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.markdown("---")
        st.markdown(f"**📁 {st.session_state.last_file}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pages", info["page_count"])
        with col2:
            st.metric("Sections", info["section_count"])
        with col3:
            st.metric("Chunks", info["chunk_count"])

        if info.get("cached"):
            st.caption("📦 Loaded from cache (instant)")
        else:
            st.caption("🆕 Freshly indexed")

        # Section navigator
        titles = info.get("section_titles", [])
        if titles:
            st.markdown("---")
            st.markdown("**📑 Sections — click to ask**")
            for title in titles[:40]:
                if st.button(
                    f"› {title[:48]}{'…' if len(title) > 48 else ''}",
                    key=f"sec_{title[:30]}",
                    use_container_width=True,
                ):
                    st.session_state.prefill_question = f"Explain: {title}"
                    st.rerun()

# ────────────────────────────────────────────────────────────────
# Ingest on upload with step-by-step progress
# ────────────────────────────────────────────────────────────────

if uploaded and st.session_state.last_file != uploaded.name:
    progress_container = st.empty()
    status_container   = st.empty()

    steps = {
        "parsing":    "📄 Extracting text from PDF…",
        "chunking":   "🔪 Splitting into sections & chunks…",
        "embedding":  "🧮 Embedding chunks locally…",
        "summarizing":"📝 Generating document summary…",
        "done":       "✅ Ready!",
    }

    def progress_callback(step_name, detail):
        msg = detail if step_name == "embedding" and detail else steps.get(step_name, detail)
        status_container.info(f"**{msg}**")

    try:
        file_bytes = uploaded.getvalue()
        result = ingest_pdf(
            file_bytes=file_bytes,
            embedder=embedder,
            chroma_client=chroma_client,
            groq_client=groq_client,
            progress_callback=progress_callback,
        )

        st.session_state.collection = result["collection_name"]
        st.session_state.doc_info   = result
        st.session_state.last_file  = uploaded.name
        st.session_state.messages   = []
        st.session_state.sources    = {}

        progress_container.empty()
        if result["cached"]:
            status_container.success(
                f"✅ **Already indexed** — {result['chunk_count']} chunks loaded instantly from cache."
            )
        else:
            status_container.success(
                f"✅ **Indexed {result['chunk_count']} chunks** from "
                f"{result['page_count']} pages ({result['section_count']} sections detected)."
            )
        st.rerun()

    except ValueError as e:
        status_container.error(f"❌ {e}")
        st.stop()

# ────────────────────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────────────────────

st.html("""
<div class="hero-header">
  <h1>🎓 StudyPal — AI Textbook Assistant</h1>
  <p>Ask anything about your uploaded PDF and get grounded, friendly answers with page citations.</p>
</div>
""")

# Document summary panel
if st.session_state.doc_info and st.session_state.doc_info.get("summary"):
    with st.expander("📋 Document Overview", expanded=False):
        st.html(
            f'<div class="summary-box">{st.session_state.doc_info["summary"]}</div>'
        )

# ────────────────────────────────────────────────────────────────
# Render previous messages
# ────────────────────────────────────────────────────────────────

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and idx in st.session_state.sources:
            sources = st.session_state.sources[idx]
            if sources:
                with st.expander(f"📚 {len(sources)} source(s) used"):
                    for s in sources:
                        score_color = (
                            "#34d399" if s["score"] >= 0.6 else
                            "#60a5fa" if s["score"] >= 0.4 else "#f59e0b"
                        )
                        st.html(
                            f'<div class="source-card">'
                            f'<div class="source-meta">'
                            f'📖 {s["section"]} &nbsp;·&nbsp; p. {s["page"]} &nbsp;·&nbsp; '
                            f'<span style="color:{score_color}">▲ {s["score"]:.2f} relevance</span>'
                            f'</div>'
                            f'<div class="source-excerpt">{s["text"][:450]}'
                            f'{"…" if len(s["text"]) > 450 else ""}</div>'
                            f'</div>'
                        )

# ────────────────────────────────────────────────────────────────
# Chat input
# ────────────────────────────────────────────────────────────────

if st.session_state.collection:
    # Consume any prefill from section navigator
    prefill = st.session_state.pop("prefill_question", "") or ""

    question = st.chat_input(
        "Ask anything about your textbook…",
        key="chat_input",
    ) or (prefill if prefill else None)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Build LLM history (skip the just-added user msg)
        llm_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ][-8:]

        doc_mode = st.session_state.doc_info.get("mode", "chunked")

        with st.chat_message("assistant"):
            # ── Meta-question: route to cached summary ────────────────
            if doc_mode != "full_context" and is_meta_question(question) and \
                    st.session_state.doc_info.get("summary"):
                with st.spinner("Looking that up in the document overview…"):
                    reply = answer_from_summary(
                        question,
                        st.session_state.doc_info["summary"],
                        groq_client,
                    )
                st.markdown(reply)
                sources = []

            else:
                # ── Streaming answer ──────────────────────────────────
                sources = []
                reply_tokens: list[str] = []
                placeholder = st.empty()

                # Small "thinking" notice while query expansion runs
                with st.spinner("🧠 Understanding your question…"):
                    gen = answer_stream(
                        question=question,
                        collection_name=st.session_state.collection,
                        embedder=embedder,
                        chroma_client=chroma_client,
                        groq_client=groq_client,
                        history=llm_history,
                        mode=doc_mode,
                        full_text=st.session_state.doc_info.get("full_text", ""),
                        query_expand=query_expand,
                    )
                    # Consume the first event (chunks) before leaving spinner
                    first_event = next(gen)
                    if first_event[0] == "chunks":
                        sources = first_event[1]

                # Stream tokens into the placeholder
                for event_type, payload in gen:
                    if event_type == "token":
                        reply_tokens.append(payload)
                        placeholder.markdown("".join(reply_tokens) + "▌")
                    elif event_type == "done":
                        break

                reply = "".join(reply_tokens)
                placeholder.markdown(reply)

                # Show sources
                if sources:
                    with st.expander(f"📚 {len(sources)} source(s) used"):
                        for s in sources:
                            score_color = (
                                "#34d399" if s["score"] >= 0.6 else
                                "#60a5fa" if s["score"] >= 0.4 else "#f59e0b"
                            )
                            st.html(
                                f'<div class="source-card">'
                                f'<div class="source-meta">'
                                f'📖 {s["section"]} &nbsp;·&nbsp; p. {s["page"]} &nbsp;·&nbsp; '
                                f'<span style="color:{score_color}">▲ {s["score"]:.2f} relevance</span>'
                                f'</div>'
                                f'<div class="source-excerpt">{s["text"][:450]}'
                                f'{"…" if len(s["text"]) > 450 else ""}</div>'
                                f'</div>'
                            )

        # Persist
        assistant_idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.sources[assistant_idx] = sources

elif not uploaded:
    # ── Landing state ──────────────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.html("""
        <div class="landing-card">
          <div class="icon">📄</div>
          <h3>Upload your textbook</h3>
          <p>Drop any PDF in the sidebar — textbooks, lecture notes, research papers. Up to 1000+ pages.</p>
        </div>""")
    with col2:
        st.html("""
        <div class="landing-card">
          <div class="icon">💬</div>
          <h3>Ask naturally</h3>
          <p>Ask in plain English. StudyPal understands your intent and searches across the whole book.</p>
        </div>""")
    with col3:
        st.html("""
        <div class="landing-card">
          <div class="icon">📍</div>
          <h3>Cited answers</h3>
          <p>Every answer shows the exact section, page number, and relevance score — no hallucinations.</p>
        </div>""")

    st.html("<br>")
    st.caption(
        "Powered by **Groq** (llama-3.3-70b-versatile) · "
        "**BGE-small** embeddings · **ChromaDB** vector store · "
        "**BM25** re-ranking · **Query expansion**"
    )
