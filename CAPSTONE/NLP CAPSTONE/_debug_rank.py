"""Debug ranking: exact cosine search vs HNSW, and dilution check."""
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

c = chromadb.PersistentClient(path="chroma_db")
m = SentenceTransformer("all-MiniLM-L6-v2")
coll = c.get_collection("doc_f1e4e6d6c2e35c4b")

# Which collection is the new PDF? Use the one created last by hash —
# just probe all collections and pick the 1040-count one.
for x in c.list_collections():
    cc = c.get_collection(x.name)
    if cc.count() == 1040:
        coll = cc
        break
print("coll:", coll.name, coll.count())

q = m.encode(["What does the book say about Raft leader election?"],
             normalize_embeddings=True)[0]

# Raw MiniLM sanity: query vs the pure Distributed-Systems body sentence
body = ("Consensus protocols like Raft elect leaders and replicate logs; "
        "Lamport clocks order events and CAP limits partitions.")
bv = m.encode([body], normalize_embeddings=True)[0]
print("sim(query, pure body sentence):", round(float(np.dot(q, bv)), 3))

# Exact search over ALL stored embeddings
data = coll.get(include=["embeddings", "metadatas"])
embs = np.array(data["embeddings"])
meta = data["metadatas"]
sims = embs @ q
order = np.argsort(-sims)[:5]
print("\nEXACT top-5:")
for i in order:
    print(f"  {sims[i]:.3f} {meta[i]['section']} p.{meta[i]['page']}")

# HNSW result
res = coll.query(query_embeddings=[q.tolist()], n_results=5,
                 include=["metadatas", "distances"])
print("\nHNSW top-5:")
for md, d in zip(res["metadatas"][0], res["distances"][0]):
    print(f"  {1 - d:.3f} {md['section']} p.{md['page']}")
