import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

from ingest import ingest_pdf, file_hash
from rag import answer, is_meta_question, answer_from_summary

load_dotenv()

TOPIC_NAMES = [
    "Process Management", "Virtual Memory", "File Systems", "Deadlocks",
    "Synchronization", "Storage and IO", "Security", "Networking",
    "Distributed Systems", "Cloud Computing",
]


def main():
    with open("data/synth_1000.pdf", "rb") as f:
        data = f.read()
    print(f"[setup] {len(data) // 1024} KB PDF")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    groq_client = Groq()

    name = f"doc_{file_hash(data)}"
    if name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name)  # force fresh ingest for timing
        print("[setup] Deleted old collection for fresh timing run")

    last = {"d": ""}
    def cb(step, detail):
        msg = detail if step == "embedding" and detail else step
        if msg != last["d"]:
            last["d"] = msg
            print(f"   [progress] {msg}")

    print("\n=== CHECK 1: INGEST with visible progress ===")
    t0 = time.time()
    result = ingest_pdf(data, embedder, chroma_client, groq_client, progress_callback=cb)
    total_s = time.time() - t0
    print(f"RESULT chunks={result['chunk_count']} pages={result['page_count']} "
          f"sections={result['section_count']} cached={result['cached']}")
    print(f"RESULT total_time={total_s:.1f}s  "
          f"overall={result['chunk_count'] / total_s:.1f} chunks/sec (incl parse+summary)")

    coll = chroma_client.get_collection(result["collection_name"])
    print(f"Collection count: {coll.count()}  (large-doc TOP_K=9: {coll.count() > 500})")
    print(f"\nSummary sample (breadth check): {result['summary'][:300]}")

    history = []
    def ask(q):
        nonlocal history
        print(f"\nQ: {q}")
        t = time.time()
        reply, srcs = answer(q, result["collection_name"], embedder,
                             chroma_client, groq_client, history)
        print(f"  ({time.time() - t:.1f}s, {len(srcs)} sources) {reply[:300]}")
        if srcs:
            print("  top src:", srcs[0]["section"], "p.", srcs[0]["page"],
                  "sim", srcs[0]["score"])
        history = (history + [{"role": "user", "content": q},
                              {"role": "assistant", "content": reply}])[-8:]
        return reply

    print("\n=== CHECK 2: broad question reflects whole book ===")
    r1 = ask("What is this book about?")
    breadth = sum(1 for t in TOPIC_NAMES if t.lower() in r1.lower())
    print(f"  -> topics mentioned: {breadth}/10")

    print("\n=== CHECK 3: deep fact (Distributed Systems ~pages 880-940) ===")
    r2 = ask("What does the book say about Raft leader election?")
    ok3 = any(w in r2.lower() for w in ("raft", "consensus", "leader"))
    print(f"  -> deep-fact retrieved correctly: {ok3}")

    print("\n=== CHECK 4: re-upload cache ===")
    t = time.time()
    r = ingest_pdf(data, embedder, chroma_client, groq_client)
    print(f"  re-upload: cached={r['cached']} in {time.time() - t:.2f}s")

    print("\n=== CHECK 5: meta routing in chunked mode ===")
    q = "summarize this book"
    print(f"  is_meta_question('{q}') = {is_meta_question(q)}")
    if is_meta_question(q):
        t = time.time()
        s = answer_from_summary(q, result["summary"], groq_client)
        print(f"  summary answer ({time.time() - t:.1f}s): {s[:250]}")


if __name__ == "__main__":
    main()
