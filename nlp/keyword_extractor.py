"""Rank JD keywords with TF-IDF, n-gram frequency and skill importance."""
from collections import Counter
from nlp.preprocessing import preprocess, ngrams

def extract_keywords(text: str, tfidf_terms: list[str], skills: list[str], required_skills: list[str], limit: int = 20) -> list[str]:
    tokens=preprocess(text).tokens
    frequencies=Counter(tokens)
    for size in (2,3): frequencies.update(ngrams(tokens,size))
    tfidf={term.lower():1.25 for term in tfidf_terms}
    required={term.lower():2.5 for term in required_skills}
    skill_boost={term.lower():1.5 for term in skills}
    for skill in skills: frequencies.setdefault(skill.lower(),1)
    ranked=sorted(frequencies, key=lambda term:(frequencies[term]+tfidf.get(term,0)+required.get(term,0)+skill_boost.get(term,0),frequencies[term],term), reverse=True)
    return ranked[:limit]
