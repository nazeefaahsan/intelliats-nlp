"""Weighted compatibility score built from separate signals."""
def score_analysis(semantic, tfidf, skill_score, keyword_score, structure_score):
    weights = {"semantic": .30, "tfidf": .22, "skills": .28, "keywords": .12, "structure": .08}
    values = {"semantic": semantic, "tfidf": tfidf, "skills": skill_score, "keywords": keyword_score, "structure": structure_score}
    if semantic is None:
        weights = {"tfidf": .32, "skills": .40, "keywords": .18, "structure": .10}
        values.pop("semantic")
    return round(sum(values[k] * w for k,w in weights.items()), 1), weights
