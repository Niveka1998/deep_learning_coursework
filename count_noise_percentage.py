import json
import re
from collections import Counter
from preprocess import clean_text

# same checks as before, minus the ones that stop making sense after cleaning
checks = {
    "empty text":      lambda t, title: not t.strip(),
    "has tags":        lambda t, title: "<s>" in t or "</s>" in t,
    "glued sections":  lambda t, title: "</s><s>" in t,
    "repeated title":  lambda t, title: t.replace("<s>", "").replace("</s>", "").strip().startswith(f"{title} {title}"),
    "heading junk .:": lambda t, title: ".:" in t,
    "empty quotes":    lambda t, title: '""' in t,
    "list-like":       lambda t, title: t.count(" - ") >= 3,
    "odd whitespace":  lambda t, title: "\n" in t or "\t" in t or "  " in t,
    "reference-style initials": lambda t, title: len(re.findall(r"\b[A-Z]\. ?[A-Z]\.", t)) >= 3,
    "curly apostrophe": lambda t, title: "\u2019" in t,
    "straight apostrophe": lambda t, title: "'" in t,
}

before, after = Counter(), Counter()
examples = {name: [] for name in checks}
cache = {}          # clean each distinct (title, passage) only once
total = 0

with open("data/passages_bm25_ChatGPT_humfacts.jsonl", encoding="utf-8") as f:
    for line in f:
        for p in json.loads(line)["passages"]:
            total += 1
            raw, title = p["text"], p["title"]
            key = (title, raw)                        #  title is part of the key
            if key not in cache:
                cache[key] = clean_text(raw, title)   #  pass the title
            cleaned = cache[key]
            for name, check in checks.items():
                if check(raw, title):
                    before[name] += 1
                if check(cleaned, title):
                    after[name] += 1
                    if len(examples[name]) < 3:
                        examples[name].append(cleaned[:200])

print("passage appearances:", total, "| distinct passages:", len(cache))
print(f"{'check':28}{'before':>8}{'after':>8}")
for name in checks:
    print(f"{name:28}{before[name]:8}{after[name]:8}")

# for rows that should now be 0, show what is still matching
for name in ["has tags", "glued sections", "heading junk .:", "odd whitespace",
             "repeated title", "empty quotes", "curly apostrophe"]:
    if after[name]:
        print("\nSTILL MATCHING:", name)
        for e in examples[name]:
            print("  ", repr(e))

# sanity check: did cleaning change any words it shouldn't?
def words(t):
    return re.findall(r"\w+", re.sub(r"</?s>", " ", t))

# (a) every rule except the title rule must keep the words identical -> expect 0
bad = [raw for (title, raw) in cache if words(raw) != words(clean_text(raw))]
print("\nwords changed by the non-title rules (expect 0):", len(bad))
for raw in bad[:3]:
    print("  ", repr(raw[:150]))

# (b) the title rule is supposed to remove words -> count should be close to the repeated-title row
title_removed = sum(1 for (title, raw), cleaned in cache.items() if cleaned != clean_text(raw))
print("distinct passages where the repeated title was removed:", title_removed)