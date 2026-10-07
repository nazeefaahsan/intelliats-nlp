"""Text normalization that preserves common technical terms."""
import re
from dataclasses import dataclass
from functools import lru_cache
STOP_WORDS = set("a an the and or but if while with to from of in on for by is are was were be been being this that these those it its as at into about our your their you we they have has had will can should must may such also".split())
TECH = {"c++": "cplusplus", "c#": "csharp", ".net": "dotnet", "node.js": "nodejs", "react.js": "reactjs", "vue.js": "vuejs", "next.js": "nextjs", "scikit-learn": "scikitlearn"}
@lru_cache(maxsize=1)
def _lemmatizer():
    """Load spaCy once when its English model is installed; otherwise use rules."""
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["textcat"])
    except Exception:
        return None
@lru_cache(maxsize=1)
def _fast_lemmatizer():
    """A lighter spaCy pipeline for normalization; parsing/NER are unnecessary."""
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["textcat", "parser", "ner"])
    except Exception:
        return None
@dataclass
class ProcessedText:
    original_text: str
    normalized_text: str
    tokens: list[str]
    sentences: list[str]
    lemmas: list[str]
def preprocess(text: str) -> ProcessedText:
    original = text or ""
    clean = re.sub(r"\s+", " ", original).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    protected = clean.lower()
    for term, token in TECH.items(): protected = re.sub(re.escape(term), token, protected, flags=re.I)
    tokens = re.findall(r"[a-z0-9]+(?:['+#.]?[a-z0-9+#]+)*", protected)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    model = _fast_lemmatizer()
    if model:
        lemmas = [word.lemma_.lower() if word.lemma_ else word.text for word in model(" ".join(tokens))]
    else:
        lemmas = [t[:-3] + "y" if t.endswith("ies") and len(t) > 4 else t[:-1] if t.endswith("s") and not t.endswith(("ss", "us", "is")) and len(t) > 3 else t for t in tokens]
    return ProcessedText(original, " ".join(lemmas), tokens, sentences, lemmas)
def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)) if not any(w in STOP_WORDS for w in tokens[i:i+n])]
