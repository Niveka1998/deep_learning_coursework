#pruning table + experiment grid (no model needed)
import csv
import json
from preprocess import candidate_units, jaccard      # swap jaccard for Member 1's overlap_score later

PASSAGES = "data/passages_bm25_ChatGPT_humfacts.jsonl"
LABELS = "data/dev_labeled_ChatGPT.jsonl"
CUTOFFS = [0.0, 0.05, 0.1, 0.15, 0.2]
TOP_K = 3

#labels
labels = {}
with open(LABELS, encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        if i == 0:
            first = next((a for a in (d.get("annotations") or []) if a.get("human-atomic-facts")), None)
            print("labels file keys:", list(d.keys()))
            print("first annotation keys:", list(first.keys()) if first else None)
            print("first fact:", first["human-atomic-facts"][0] if first else None, "\n")
        for ann in d.get("annotations") or []:
            for fact in ann.get("human-atomic-facts") or []:
                labels[(d["topic"], fact["text"])] = fact["label"]      # 'S' / 'NS' / 'IR'
print("labelled facts loaded:", len(labels), "| label counts:",
      {l: list(labels.values()).count(l) for l in set(labels.values())})

items = []
with open(PASSAGES, encoding="utf-8") as f:
    for line in f:
        it = json.loads(line)
        items.append((it["name"], it["sent"], it["passages"]))

total_lines = len(items)
items = [x for x in items if (x[0], x[1]) in labels]     # only the labelled dev facts
n_s = sum(labels[(n, s)] == "S" for n, s, _ in items)

print(f"passage lines: {total_lines} | kept (have a label): {len(items)} "
      f"| labelled facts in labels file: {len(labels)}")
print(f"S facts: {n_s}")
if len(items) != len(labels):
    print("NOTE: kept lines != labelled facts, so some labels have no passage line or a fact is duplicated")

# (name, kwargs for candidate_units, cutoffs)
CONFIGS = [
    ("whole passage, raw",          dict(window=0, clean=False), [0.0]),
    ("whole passage, cleaned",      dict(window=0, clean=True),  [0.0]),
    ("1 sentence, raw",             dict(window=1, clean=False), CUTOFFS),
    ("1 sentence, cleaned",         dict(window=1, clean=True),  CUTOFFS),
    ("1 sent, clean, drop lists",   dict(window=1, clean=True, drop_lists=True), CUTOFFS),
    ("1 sent, clean, min 2 words",  dict(window=1, clean=True, min_words=2), CUTOFFS),
    ("1 sent, clean, min 5 words",  dict(window=1, clean=True, min_words=5), CUTOFFS),
    ("2-sentence window, clean",    dict(window=2, clean=True), CUTOFFS),
    ("3-sentence window, clean",    dict(window=3, clean=True), CUTOFFS),
]

rows = []
for name, kw, cutoffs in CONFIGS:
    per_fact = []
    for n, fact, passages in items:
        units = candidate_units(passages, **kw)
        per_fact.append((labels[(n, fact)],
                         [jaccard(fact, u) for u in units],
                         [len(u.split()) for u in units]))
    total_units = sum(len(sc) for _, sc, _ in per_fact)

    for c in cutoffs:
        kept = sum(sum(s >= c for s in sc) for _, sc, _ in per_fact)
        words_kept = sum(l for _, sc, ln in per_fact for s, l in zip(sc, ln) if s >= c)
        s_lost = sum(lab == "S" and sc and not any(s >= c for s in sc) for lab, sc, _ in per_fact)
        ns_lost = sum(lab == "NS" and sc and not any(s >= c for s in sc) for lab, sc, _ in per_fact)
        after_guard = sum((sum(s >= c for s in sc) or min(TOP_K, len(sc))) for _, sc, _ in per_fact)
        rows.append({
            "config": name, "cutoff": c,
            "avg_premises_before": round(total_units / len(items), 2),
            "avg_premises_after_cutoff": round(kept / len(items), 2),
            "avg_premises_sent_to_model": round(after_guard / len(items), 2),
            "avg_words_per_fact": round(words_kept / len(items)),
            "pct_units_kept": round(100 * kept / max(1, total_units), 1),
            "S_facts_losing_all": s_lost,
            "S_facts_losing_all_pct": round(100 * s_lost / max(1, n_s), 1),
            "NS_facts_losing_all": ns_lost,
            "acc_word_overlap": "",
            "acc_entailment": "",
        })
    print(name, "done")

with open("experiment_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n{'config':28}{'cut':>6}{'before':>8}{'after':>8}{'to model':>9}{'words':>7}{'%kept':>7}{'S lost':>8}{'S lost%':>9}")
for r in rows:
    print(f"{r['config']:28}{r['cutoff']:6}{r['avg_premises_before']:8}{r['avg_premises_after_cutoff']:8}"
          f"{r['avg_premises_sent_to_model']:9}{r['avg_words_per_fact']:7}{r['pct_units_kept']:7}"
          f"{r['S_facts_losing_all']:8}{r['S_facts_losing_all_pct']:9}")
print("\nsaved experiment_results.csv")