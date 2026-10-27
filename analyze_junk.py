# A .jsonl file has one JSON per line. So we read it line by line and turn each line into a python dictionary with json.loads.

import json

seen = set()
with open("data/passages_bm25_ChatGPT_humfacts.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        name = item["name"]

        if name in seen:
            continue
        seen.add(name)

        print(name)
        if item["passages"]:
            print(repr(item["passages"][0]["text"][:400]))
        else:
            print("No Passages")
        print("--------------")
        if len(seen) ==10:
            break


