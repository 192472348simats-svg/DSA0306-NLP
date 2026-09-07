"""Verify hnsw:search_ef tuning fixes recall on the 1040-chunk collection."""
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

c = chromadb.PersistentClient(path="chroma_db")
m = SentenceTransformer("all-MiniLM-L6-v2")

src = None
for x in c.list_collections():
    cc = c.get_collection(x.name)
    if cc.count() == 1040:
        src = cc
        break

data = src.get(include=["embeddings", "documents", "metadatas"])
print("source:", src.name, src.count())

q = m.encode(["What does the book say about Raft leader election?"],
             normalize_embeddings=True)[0]

for ef in (100, 200, 400):
    name = f"tune-ef{ef}"
    try:
        c.delete_collection(name)
    except Exception:
        pass
    coll = c.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine",
                  "hnsw:construction_ef": ef,
                  "hnsw:search_ef": ef},
    )
    B = 256
    for i in range(0, len(data["embeddings"]), B):
        coll.add(
            ids=data["ids"][i:i+B],
            embeddings=data["embeddings"][i:i+B],
            documents=data["documents"][i:i+B],
            metadatas=data["metadatas"][i:i+B],
        )
    res = coll.query(query_embeddings=[q.tolist()], n_results=5,
                     include=["metadatas", "distances"])
    print(f"\nsearch_ef={ef}:")
    for md, d in zip(res["metadatas"][0], res["distances"][0]):
        print(f"  {1 - d:.3f} {md['section']} p.{md['page']}")
    c.delete_collection(name)
