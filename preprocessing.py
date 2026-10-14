import re
from nltk.tokenize import sent_tokenize

def clean_text(text: str) -> str:
    """Remove wikipedia conversation noise from a raw passage"""
    # one re.sub per noise row in report/noise_table.md
    return text.strip()

def split_sentences(text: str, min_words: int=3) -> list[str]:
    """Split a cleaned passage into sentences, merging fragments"""
    #todo
    return 0.0

def prepare_pairs(fact:str, passages: list[dict]) -> list[str]:
    """Cleaned, sentence-split, pruned premise sentences for one fact"""
    #todo
    return[]

def _jaccard_overlap(fact: str, text: str) -> float:
    """TEMPORARY stand-in (Jaccard on lowercase word sets).
    Same i/o as member 1's overlap_score(fact, text) -> float."""
    a = set(re.findall(r"\w+", fact.lower()))
    b = set(re.findall(r"\w+", text.lower()))
    if not a and not b:
        return 0.0
    return len(a&b) / len(a|b)

#temporary
print(_jaccard_overlap("Marie Curie won a prize", "Marie Curie won a prize"))   # 1.0
print(_jaccard_overlap("Marie Curie won a prize", "The cat sat on the mat"))    # 0.0
print(_jaccard_overlap("Marie Curie won a prize", "Curie won the Nobel Prize in 1903"))  # between 0 and 1