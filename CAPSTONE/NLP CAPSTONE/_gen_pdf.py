import fitz

TOPICS = [
    ("Process Management", "The scheduler picks ready processes using priority queues and context switches save register state in the PCB."),
    ("Virtual Memory", "Page tables map virtual pages to physical frames; the TLB caches recent translations and demand paging loads pages on fault."),
    ("File Systems", "Inodes store file metadata and block pointers; journaling guarantees crash consistency and bitmaps track free space."),
    ("Deadlocks", "Deadlock requires mutual exclusion, hold-and-wait, no preemption and circular wait; Banker's algorithm avoids unsafe states."),
    ("Synchronization", "Semaphores and monitors enforce mutual exclusion; test-and-set provides atomic locks and condition variables coordinate waits."),
    ("Storage and IO", "Disk scheduling uses SSTF, SCAN and C-SCAN; RAID levels trade redundancy for performance and buffers smooth transfers."),
    ("Security", "Access control lists authorize operations; salted hashes protect passwords and ASLR with stack canaries stop overflows."),
    ("Networking", "TCP provides reliable byte streams with congestion control; routing algorithms compute shortest paths via sockets."),
    ("Distributed Systems", "Consensus protocols like Raft elect leaders and replicate logs; Lamport clocks order events and CAP limits partitions."),
    ("Cloud Computing", "Virtualization multiplexes hardware across tenants; containers share the host kernel and autoscaling adjusts capacity."),
]
PARA = ("This section explains core mechanisms in detail with worked examples "
        "and review questions. Performance depends on workload characteristics. "
        "Design trade-offs balance simplicity, correctness and speed. ")

doc = fitz.open()
n = 0
for ci, (t, b) in enumerate(TOPICS, 1):
    for si in range(1, 9):
        for _ in range(13):
            n += 1
            # Realistic page: header + topic body + unique per-page facts
            txt = (f"{ci}.{si} {t} Part {si}\n\n{b}\n\n{PARA * 6}\n"
                   f"Fact: {t} detail {si} on page {n}. "
                   f"Review: explain {t.lower()} and how it differs from "
                   f"related mechanisms covered in chapter {ci}.\n")
            pg = doc.new_page()
            pg.insert_text((50, 72), txt[:3000], fontsize=10)
doc.save("data/synth_1000.pdf")
doc.close()
print("pages:", n)
