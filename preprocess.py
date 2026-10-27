# holds the functions -cleaning, sentence splitting, pruning
import re
from functools import lru_cache

#cleaning
def strip_repeated_title(text, title):
    """'Joeri Adams Joeri Adams (born 15 ' -> 'Joeri Adams (born 15 '"""
    bases = [title, re.sub(r"\s*\([^)]*\)\s*$", "", title)]
    for t in dict.fromkeys(b for b in bases if b):
        if text.startswith(f"{t} {t}"):
            return text[len(t) + 1:]
    return text


def clean_text(text, title=None):
    #rule 1: if the text is empty, return an empty string
    if not text:
        return ""

    #rule 2: curly quotes -> straight quotes/apostrophes (so the tokenizer sees one style)
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u201c", '"').replace("\u201d", '"'))


    #rule 3: tags
    #3.1 a section boundary </s><s> after text with no end punctuation becomes a full stop
    text = re.sub(r"([\w)\]]\"?)\s*</s>\s*<s>", r"\1. ", text)

    #3.2 every remaining tag becomes a space (extra spaces are tidied by the whitespace rule later)
    text = re.sub(r"</?s>", " ", text)

    #rule 4: heading junk '.:' ('Major results.:Cyclo-cross.' -> 'Major results. Cyclo-cross')
    text = text.replace(".:", ". ")

    #rule 5: rempty quotes: exactly two quote marks, not part of a longer run
    text = re.sub(r'(?<!")""(?!")', " ", text)

    # rule 6: whitespace
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,;])", r"\1", text)  # ' ,' left behind by removed quotes

    # rule 7: repeated title (needs the title, runs after whitespace is clean)
    if title:
        text = strip_repeated_title(text, title)
    return text

# sentence splitting, lists, windows, length
_warned = False
def split_sentences(text, min_words=3):
    """nltk sent_tokenize, then drop pieces shorter than min_words
        (this is what removes heading fragments like 'Career.')."""
    global _warned
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize
        pieces = sent_tokenize(text)
    except (ImportError, LookupError):
        # run nltk.download('punkt_tab') once; this fallback is only so the code never crashes
        if not _warned:
            print("WARNING: nltk punkt not available, using regex splitter")
            _warned = True
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])", text)
    return [s.strip() for s in pieces if len(s.split()) >= min_words]


def is_list_like(s):
    """same rule as the noise count: 3+ ' - ' separators."""
    return s.count(" - ") >= 3


def make_windows(sents, window):
    """window 1 -> sentences as they are. window 2/3 -> sliding windows of
    neighbouring sentences joined with a space."""
    if not sents:
        return []
    if window <= 1:
        return list(sents)
    if len(sents) <= window:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + window]) for i in range(len(sents) - window + 1)]


def truncate_words(s, max_words):
    """Word-based guard so a premise + fact stays under the 512-token limit.
    350 words is roughly 450 subword tokens. Tuned with measure_sentences.py."""
    w = s.split()
    return s if len(w) <= max_words else " ".join(w[:max_words])

#stand-in scorer
@lru_cache(maxsize=None)
def _tokens(text):
    return frozenset(re.findall(r"\w+", text.lower()))

def jaccard(fact, text):
    a,b = _tokens(fact), _tokens(text)
    return len(a&b) / len(a|b) if a and b else 0.0

#prepare pairs

_units_cache = {} #one entry per distinct passage + settings
def _passage_units(p, clean, min_words, drop_lists, window, max_words):
    key = (p["title"], p["text"], clean, min_words, drop_lists, window, max_words)
    if key not in _units_cache:
        text = clean_text(p["text"], p["title"]) if clean else p["text"]
        if window == 0:                       # baseline: whole passage as one premise
            units = [text] if text else []
        else:
            sents = split_sentences(text, min_words)
            if drop_lists:
                sents = [s for s in sents if not is_list_like(s)]
            units = make_windows(sents, window)
        _units_cache[key] = [truncate_words(u, max_words) for u in units]
    return _units_cache[key]

def candidate_units(passages, window=1, clean=True, min_words=3,
                    drop_lists=False, max_words=350):
    """All premise candidates for one fact's passages, de-duplicated, in order.
    (Retrieved chunks can overlap, so the same sentence may appear twice.)"""
    units, seen = [], set()
    for p in passages:
        for u in _passage_units(p, clean, min_words, drop_lists, window, max_words):
            if u not in seen:
                seen.add(u)
                units.append(u)
    return units

def prepare_pairs(fact, passages, cutoff=0.1, window=1, clean=True, min_words=3,
                  drop_lists=False, scorer=jaccard, top_k=3, max_words=350):
    """-> list[str]: cleaned, sentence-split, pruned premises for ONE fact.
    Safeguard: if the cutoff removes everything, keep the top_k best-scoring ones."""
    units = candidate_units(passages, window, clean, min_words, drop_lists, max_words)
    if not units:
        return []
    scored = [(scorer(fact, u), i, u) for i, u in enumerate(units)]
    kept = [x for x in scored if x[0] >= cutoff]
    if not kept:
        kept = sorted(scored, key=lambda x: (-x[0], x[1]))[:top_k]
        kept.sort(key=lambda x: x[1])          # restore passage order
    return [u for _, _, u in kept]

#tests
if __name__ == "__main__":
    tests = [
        ("", ""),
        (None, ""),
        ("   ", ""),
        ("A normal clean sentence.", "A normal clean sentence."),
        ("  spaces  and\ttabs\n", "spaces and tabs"),
        ("<s>tags stay</s>", "tags stay"),
        ("in 2017.</s><s>Early life.", "in 2017. Early life."),
        ("<s>Keita Kadokura</s><s>Early life.</s>", "Keita Kadokura. Early life."),
        ("first,</s><s>second", "first, second"),
        ("Works:</s><s>List", "Works: List"),
        ('said."</s><s>Next', 'said." Next'),
        ("(born 1970)</s><s>Next", "(born 1970). Next"),
        ("tail of a list</s>", "tail of a list"),
        ("No tags at all.", "No tags at all."),
        ('iscient"</s><s>Discography. - (Haagensen', 'iscient". Discography. - (Haagensen'),
        ('total)"</s><s>Honours. Am', 'total)". Honours. Am'),
        ('feelings."</s><s>Career.', 'feelings." Career.'),
        ("Major results.:Cyclo-cross.", "Major results. Cyclo-cross."),
        ("Career.</s><s>Career.:2008–2009: Career beginnings.", "Career. Career. 2008–2009: Career beginnings."),
        # new rules
        ('"Natural Born Killers", "" and "Signs".', '"Natural Born Killers", and "Signs".'),
        ('He said "" today.', "He said today."),
        ('A "quoted" word.', 'A "quoted" word.'),
        ("UNAM\u2019s campus", "UNAM's campus"),
    ]
    bad = 0
    for before, expected in tests:
        after = clean_text(before)
        ok = after == expected
        bad += not ok
        print("OK  " if ok else "FAIL", repr(before), "->", repr(after))

    # title rule (needs the title argument)
    title_tests = [
        ("<s>Taral Hicks Taral Hicks (born 1974) is an actress.</s>", "Taral Hicks",
         "Taral Hicks (born 1974) is an actress."),
        ("Gerhard Fischer Gerhard Fischer was an inventor.", "Gerhard Fischer (inventor)",
         "Gerhard Fischer was an inventor."),
        ("Taral Hicks is an actress.", "Taral Hicks", "Taral Hicks is an actress."),
    ]
    for before, title, expected in title_tests:
        after = clean_text(before, title)
        ok = after == expected
        bad += not ok
        print("OK  " if ok else "FAIL", "title:", repr(after))

    # windows / scorer / prepare_pairs
    assert make_windows(["a b c", "d e f", "g h i"], 2) == ["a b c d e f", "d e f g h i"]
    assert make_windows(["a b c"], 3) == ["a b c"]
    assert jaccard("a b c", "a b c") == 1.0 and jaccard("", "x") == 0.0
    passages = [{"title": "Lanny Flaherty",
                 "text": "<s>Lanny Flaherty Lanny Flaherty (born July 27, 1942) is an American actor.</s>"
                         "<s>Career. He appeared in many films and television shows over four decades.</s>"}]
    out = prepare_pairs("Lanny Flaherty is an American actor.", passages, cutoff=0.3)
    print(out)
    assert len(out) == 1 and "American actor" in out[0]
    # safeguard: impossible cutoff still returns something
    assert len(prepare_pairs("zzz qqq", passages, cutoff=0.9)) >= 1
    print("ALL PASSED" if not bad else f"{bad} FAILED")