"""Lexical similarity and interpretable TF-IDF terms."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def compare(resume: str, jd: str) -> dict:
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=2500, stop_words="english")
        # Keep work bounded for unusually text-heavy PDFs.
        matrix = vectorizer.fit_transform([(resume or " ")[:50000], (jd or " ")[:50000]])
        score = float(cosine_similarity(matrix[0], matrix[1])[0,0] * 100)
        terms = vectorizer.get_feature_names_out(); weights = matrix[1].toarray().ravel()
        return {"score": round(score,1), "top_terms": [terms[i] for i in weights.argsort()[::-1] if weights[i] > 0][:20]}
    except ValueError: return {"score": 0.0, "top_terms": []}
