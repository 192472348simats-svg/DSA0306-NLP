"""
eval.py — Offline evaluation of the RAG pipeline.

Run separately (NOT in the chatbot's request path):
    python eval.py

Usage:
  1. Place a test PDF at data/os_textbook.pdf (or change TEST_PDF below).
  2. Fill in TEST_CASES with hand-written Q&A pairs for that PDF.
  3. Run: python eval.py
"""

import os
import sys
import time
import re
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from ingest import ingest_pdf
from rag import retrieve, answer

load_dotenv()

TEST_PDF = "data/os_textbook.pdf"

# ── Initialize clients ──
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

TEST_CASES = [
    {
        "question": "What is a deadlock?",
        "expected_keywords": ["deadlock", "wait", "resource", "process"],
        "expected_sections": [],
        "expected_answer": "",
    },
    {
        "question": "Explain the difference between paging and segmentation.",
        "expected_keywords": ["paging", "segmentation", "page", "segment", "memory"],
        "expected_sections": [],
        "expected_answer": "",
    },
    {
        "question": "What are the four conditions for deadlock?",
        "expected_keywords": [
            "mutual exclusion", "hold and wait",
            "no preemption", "circular wait",
        ],
        "expected_sections": [],
        "expected_answer": "",
    },
    {
        "question": "How does round-robin scheduling work?",
        "expected_keywords": ["round-robin", "time quantum", "preempt", "queue"],
        "expected_sections": [],
        "expected_answer": "",
    },
    {
        "question": "What is virtual memory?",
        "expected_keywords": ["virtual", "memory", "page", "address", "physical"],
        "expected_sections": [],
        "expected_answer": "",
    },
]


def keyword_overlap_score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def section_recall(retrieved_chunks: list[dict], expected_sections: list[str]) -> float:
    if not expected_sections:
        return -1.0
    retrieved_secs = {c["section"] for c in retrieved_chunks}
    hits = sum(
        1 for sec in expected_sections
        if any(sec.lower() in rs.lower() for rs in retrieved_secs)
    )
    return hits / len(expected_sections)


def exact_match(answer_text: str, expected: str) -> bool:
    if not expected:
        return False
    normalise = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    return normalise(expected) in normalise(answer_text)


def run_evaluation():
    if not os.path.isfile(TEST_PDF):
        print(f"❌  Test PDF not found at '{TEST_PDF}'.")
        sys.exit(1)

    print(f"📖 Ingesting '{TEST_PDF}' …")
    with open(TEST_PDF, "rb") as f:
        file_bytes = f.read()
    result = ingest_pdf(file_bytes, embedder, chroma_client, groq_client)
    collection_name = result["collection_name"]
    print(f"   Collection: {collection_name} ({result['chunk_count']} chunks)")

    print("\n" + "=" * 70)
    print("RAG Pipeline — Offline Evaluation")
    print("=" * 70)

    results = []

    for i, tc in enumerate(TEST_CASES, 1):
        q = tc["question"]
        print(f"\n{'─' * 60}")
        print(f"Q{i}: {q}")
        print(f"{'─' * 60}")

        t0 = time.time()
        chunks = retrieve(q, collection_name, embedder, chroma_client)
        retrieval_time = time.time() - t0

        t1 = time.time()
        ans, _ = answer(q, collection_name, embedder, chroma_client, groq_client)
        generation_time = time.time() - t1
        total_time = retrieval_time + generation_time

        kw_score = keyword_overlap_score(ans, tc["expected_keywords"])
        sec_recall = section_recall(chunks, tc["expected_sections"])
        em = exact_match(ans, tc["expected_answer"])

        r = {
            "question": q,
            "keyword_overlap": kw_score,
            "section_recall": sec_recall,
            "exact_match": em,
            "retrieval_ms": round(retrieval_time * 1000),
            "generation_ms": round(generation_time * 1000),
            "total_ms": round(total_time * 1000),
            "num_chunks": len(chunks),
        }
        results.append(r)

        print(f"  Answer: {ans[:200]}…")
        print(f"  Keyword overlap: {kw_score:.0%}")
        print(f"  Latency: {r['total_ms']}ms")
        print(f"  Chunks: {len(chunks)}")
        for c in chunks:
            print(f"    • {c['section']} (p. {c['page']}, sim {c['score']:.3f})")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    avg_kw = sum(r["keyword_overlap"] for r in results) / len(results)
    avg_latency = sum(r["total_ms"] for r in results) / len(results)
    print(f"  Questions: {len(results)}")
    print(f"  Avg keyword overlap: {avg_kw:.0%}")
    print(f"  Avg latency: {avg_latency:.0f}ms")


if __name__ == "__main__":
    run_evaluation()
