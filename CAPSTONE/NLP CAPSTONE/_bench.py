"""Benchmark embed batch sizes 64/128/256 on realistic 600-token chunks."""
import time
import tiktoken
from sentence_transformers import SentenceTransformer

enc = tiktoken.get_encoding("cl100k_base")
# Realistic textbook chunk: ~600 tokens of prose
para = ("The operating system manages hardware resources and provides services "
        "to application programs. Process scheduling determines which process "
        "runs next, balancing throughput and fairness. Memory management maps "
        "virtual addresses to physical frames, while file systems organize "
        "persistent storage into directories and files. ")
base = enc.encode(para)
chunk = enc.decode((base * 20)[:600])

# 20 realistic chunks * 24 variants = 480 chunks (enough for several batches)
texts = []
for v in range(24):
    texts.append(chunk + f" Section {v+1}.{v%9} variant {v} adds distinct vocabulary about algorithms, networks, and security.")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Warmup (model load + first encode)
model.encode(texts[:8], normalize_embeddings=True)

for bs in (64, 128, 256):
    t0 = time.time()
    for i in range(0, len(texts), bs):
        model.encode(texts[i:i+bs], normalize_embeddings=True)
    dt = time.time() - t0
    print(f"batch={bs:3d}: {dt:5.2f}s total, {len(texts)/dt:6.1f} chunks/sec")
