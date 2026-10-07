from nlp.preprocessing import preprocess, ngrams
from nlp.skill_extractor import extract_skills, flatten
from nlp.tfidf_matcher import compare
from nlp.keyword_extractor import extract_keywords
from nlp.jd_analyzer import analyze_jd
from nlp import semantic_matcher
from nlp.linguistic_analyzer import analyze_linguistics
from nlp.section_parser import parse_sections
from scoring.ats_score import score_analysis
from utils.pdf_parser import extract_text_from_pdf, PDFError
from io import BytesIO

def test_preprocess_keeps_space_and_technical_term():
    p=preprocess("Python and C++ projects. Building models")
    assert "cplusplus" in p.tokens and " " in p.normalized_text
    assert p.sentences
def test_spacy_lemmatization_when_model_is_installed():
    from nlp.preprocessing import _lemmatizer
    if _lemmatizer() is not None:
        assert "run" in preprocess("Running projects").lemmas
def test_ngrams(): assert "machine learning" in ngrams(["machine","learning","models"],2)
def test_trigram_keyword_extraction(): assert "natural language processing" in extract_keywords("natural language processing with Python",[],[],[])
def test_jd_requirement_strength():
    result=analyze_jd("Data Analyst Intern\nPython is required. Git is preferred.",["Python","Git"])
    assert "Python" in result["required_skills"] and "Git" in result["preferred_skills"]
def test_semantic_model_failure_falls_back(monkeypatch):
    monkeypatch.setattr(semantic_matcher,"_model",lambda: (_ for _ in ()).throw(RuntimeError("model unavailable")))
    result=semantic_matcher.compare_analysis("resume text","job text",{}, {},["Python"],["Python"])
    assert result["available"] is False and result["score"] is None
def test_linguistic_features_are_exposed():
    features=analyze_linguistics("Data scientists build useful machine learning tools. Python supports model development.")
    assert features["sentence_count"]==2
    assert "NOUN" in features["pos_counts"]
    assert features["noun_phrases"]
def test_taxonomy_skill_extraction_and_alias():
    found=flatten(extract_skills("Built NLP tools with Python, pandas and sklearn."))
    assert "Python" in found and "Natural Language Processing" in found and "Scikit-learn" in found
def test_tfidf_identical_text(): assert compare("python machine learning", "python machine learning")["score"] > 99
def test_tfidf_ranks_requirement_relevant_resume_evidence():
    result=compare("python nlp recommendation systems website design", "python natural language processing", {"experience":"Built recommendation systems using Python and natural language processing.","projects":"Designed a personal website."}, {"title":"NLP Engineer","required_skills":["Python","Natural Language Processing"],"preferred_skills":[],"responsibilities":["Build text classification systems"]})
    assert result["context_score"] is not None
    assert result["evidence"][0]["section"] == "experience"
def test_section_detection(): assert "projects" in parse_sections("Projects\nBuilt an NLP tool")
def test_score_fallback_is_weighted():
    score,weights=score_analysis(None,80,70,60,50)
    assert 0 <= score <= 100 and "semantic" not in weights
def test_invalid_pdf():
    try: extract_text_from_pdf(BytesIO(b"not pdf"))
    except PDFError: pass
    else: assert False
def test_empty_pdf():
    try: extract_text_from_pdf(BytesIO(b""))
    except PDFError as exc: assert "empty" in str(exc).lower()
    else: assert False
def test_image_only_pdf_message():
    import pymupdf as fitz
    doc=fitz.open(); doc.new_page(); raw=doc.tobytes(); doc.close()
    try: extract_text_from_pdf(BytesIO(raw))
    except PDFError as exc: assert "scanned images" in str(exc)
    else: assert False
