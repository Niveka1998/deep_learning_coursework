import json
import random

path = "data/passages_bm25_ChatGPT_humfacts.jsonl"

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("total lines:", len(rows))

random.seed(42)                          # same sample every run
indices = random.sample(range(len(rows)), 100)

seen = set()                             # passages already printed
for i in indices:
    row = rows[i]
    print(f"[line {i}] NAME: {row['name']}")
    print("FACT:", row["sent"])
    for p in row["passages"]:
        if p["text"] in seen:            # skip repeats
            continue
        seen.add(p["text"])
        print("  TITLE:", p["title"])
        print("  TEXT :", repr(p["text"]))
        print("  " + "-" * 40)