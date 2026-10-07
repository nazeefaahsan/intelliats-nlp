"""TF-IDF similarity with section-aware, sentence-level resume evidence."""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESUME_SECTION_ORDER = ("summary", "skills", "experience", "internship", "projects")


def _sentences(text: str, limit: int = 120) -> list[str]:
    candidates = []
    for line in re.split(r"\n+", text or ""):
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = re.sub(r"\s+", " ", sentence).strip(" \t-•")
            if len(sentence.split()) >= 2:
                candidates.append(sentence[:600])
                if len(candidates) >= limit:
                    return candidates
    return candidates


def _focus_text(jd: str, jd_info: dict | None) -> str:
    if not jd_info:
        return jd or ""
    parts = [jd_info.get("title", "")]
    for key in ("required_skills", "preferred_skills", "responsibilities"):
        parts.extend(jd_info.get(key, []))
    return "\n".join(part for part in parts if part) or (jd or "")


def _resume_evidence(resume: str, sections: dict | None) -> list[tuple[str, str]]:
    if sections:
        evidence = []
        for section in RESUME_SECTION_ORDER:
            evidence.extend((section, sentence) for sentence in _sentences(sections.get(section, ""), 40))
        if evidence:
            return evidence[:160]
    return [("resume", sentence) for sentence in _sentences(resume, 120)]


def compare(resume: str, jd: str, resume_sections: dict | None = None, jd_info: dict | None = None) -> dict:
    """Compare documents and rank relevant resume sentences with cosine similarity.

    The final TF-IDF score blends whole-document similarity with the mean of
    the three strongest requirement-to-resume sentence similarities. This
    keeps the score explainable and gives relevant experience more weight than
    unrelated text elsewhere in a long resume.
    """
    resume_text = (resume or " ")[:50000]
    jd_text = (jd or " ")[:50000]
    try:
        document_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=5000, stop_words="english", sublinear_tf=True
        )
        document_matrix = document_vectorizer.fit_transform([resume_text, jd_text])
        document_score = float(cosine_similarity(document_matrix[0], document_matrix[1])[0, 0] * 100)
        terms = document_vectorizer.get_feature_names_out()
        weights = document_matrix[1].toarray().ravel()
        top_terms = [terms[i] for i in weights.argsort()[::-1] if weights[i] > 0][:20]
    except ValueError:
        document_score, top_terms = 0.0, []

    focus = _focus_text(jd, jd_info)
    jd_sentences = _sentences(focus, 60)
    resume_evidence = _resume_evidence(resume_text, resume_sections)
    relevant_evidence = []
    context_score = None
    if jd_sentences and resume_evidence:
        corpus = jd_sentences + [sentence for _, sentence in resume_evidence]
        try:
            evidence_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_features=5000, stop_words="english", sublinear_tf=True
            )
            evidence_matrix = evidence_vectorizer.fit_transform(corpus)
            query = evidence_vectorizer.transform([focus])
            jd_count = len(jd_sentences)
            similarities = cosine_similarity(evidence_matrix[jd_count:], query).ravel()
            order = similarities.argsort()[::-1]
            relevant_evidence = [
                {"section": resume_evidence[i][0], "text": resume_evidence[i][1], "score": round(float(similarities[i]) * 100, 1)}
                for i in order[:5] if similarities[i] > 0
            ]
            if len(order) and similarities[order[0]] > 0:
                context_score = float(similarities[order[:3]].mean() * 100)
        except ValueError:
            pass

    score = document_score if context_score is None else .45 * document_score + .55 * context_score
    return {
        "score": round(score, 1),
        "document_score": round(document_score, 1),
        "context_score": round(context_score, 1) if context_score is not None else None,
        "top_terms": top_terms,
        "evidence": relevant_evidence,
    }
