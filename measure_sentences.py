#sentences per passage + how many units exceed the token limit

import json
import statistics
from preprocess import clean_text, split_sentences, make_windows

PASSAGES = "data/passages_bm25_ChatGPT_humfacts.jsonl"
LIMIT = 512
FACT_ROOM = 40  # tokens kept free for the fact itself
BUDGET = LIMIT - FACT_ROOM

#Rough estimate: 1 word ~1.3 subword tokens. For exact numbers, load the real tokenizer.
#   from transformers import AutoTokenizer
#   tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")   # use whatever model Member 3 uses
#   def n_tokens(s): return len(tok(s)["input_ids"])
def n_tokens(s):
    return int(len(s.split()) * 1.3)


# distinct passages only (same passage is retrieved for many facts)
distinct = {}
with open(PASSAGES, encoding="utf-8") as f:
    for line in f:
        for p in json.loads(line)["passages"]:
            distinct[(p["title"], p["text"])] = p
print("distinct passages:", len(distinct))

cleaned = [clean_text(t, title) for (title, t) in distinct]
sent_lists = [split_sentences(c) for c in cleaned]

counts = [len(s) for s in sent_lists]
print("\nsentences per passage: mean %.1f | median %d | max %d | passages with 0 sentences: %d"
      % (statistics.mean(counts), statistics.median(counts), max(counts), counts.count(0)))

print(f"\nover-length units (> {BUDGET} estimated tokens, so fact + premise would pass {LIMIT}):")
print(f"{'unit':22}{'units':>9}{'over':>8}{'% over':>9}")


def report(name, units):
    over = sum(n_tokens(u) > BUDGET for u in units)
    print(f"{name:22}{len(units):9}{over:8}{100 * over / max(1, len(units)):8.1f}%")


report("whole passage", [c for c in cleaned if c])
for w in (1, 2, 3):
    units = [u for s in sent_lists for u in make_windows(s, w)]
    report(f"window {w}", units)

# how long are the longest single sentences? (these are usually lists with no full stops)
long_ones = sorted((s for sl in sent_lists for s in sl), key=lambda s: -len(s.split()))[:3]
print("\nthree longest sentences (word counts):", [len(s.split()) for s in long_ones])
for s in long_ones:
    print("  ", repr(s[:150]))