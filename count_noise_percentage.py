import json
import re
from collections import Counter

checks = {
    "empty text":      lambda t, title: not t.strip(),
    "has tags":        lambda t, title: "<s>" in t or "</s>" in t,
    "glued sections":  lambda t, title: "</s><s>" in t,
    "cut-off start":   lambda t, title: not t.lstrip().startswith("<s>"),
    "cut-off end":     lambda t, title: not t.rstrip().endswith("</s>"),
    "repeated title":  lambda t, title: t.replace("<s>", "").replace("</s>", "").strip().startswith(f"{title} {title}"),
    "heading junk .:": lambda t, title: ".:" in t,
    "empty quotes":    lambda t, title: '""' in t,
    "list-like":       lambda t, title: t.count(" - ") >= 3,
    "odd whitespace":  lambda t, title: "\n" in t or "\t" in t or "  " in t,
    "heading before sentence": lambda t, title: bool(re.search(r"<s>[A-Z][A-Za-z ,&'-]{1,40}\. [A-Z]", t)),
    "reference-style initials": lambda t, title: len(re.findall(r"\b[A-Z]\. ?[A-Z]\.", t)) >= 3,
    "curly apostrophe ’":       lambda t, title: "’" in t,
    "straight apostrophe '":    lambda t, title: "'" in t,
}

counts = Counter()
total = 0

with open("data/passages_bm25_ChatGPT_humfacts.jsonl", encoding="utf-8") as f:
    for line in f:
        for p in json.loads(line)["passages"]:
            total += 1
            for name, check in checks.items():
                if check(p["text"], p["title"]):
                    counts[name] += 1

print("total passages:", total)
for name in checks:
    print(f"{name:18} {counts[name]:6}  {100 * counts[name] / total:5.1f}%")

#
# import json
# unique = set()
# found = 0
# with open("data/passages_bm25_ChatGPT_humfacts.jsonl", encoding="utf-8") as f:
#     for line in f:
#         item = json.loads(line)
#         for p in item["passages"]:
#             text = p["text"]
#
#             if "<s>" in text or "</s>" in text:
#                 continue
#             if text in unique:
#                 continue
#             unique.add(text)
#             print(p["title"])
#             print(repr(p["text"][:400]))
#             print("-----")
#             found += 1
#             if found == 3:
#                 break
#         if found == 3:
#             break