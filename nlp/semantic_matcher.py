"""Optional cached sentence-transformer semantic similarity."""
from functools import lru_cache
import numpy as np
@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")
def compare(resume: str, jd: str) -> dict:
    if not resume.strip() or not jd.strip(): return {"score": None, "available": False}
    try:
        vectors = _model().encode([resume[:6000], jd[:6000]], normalize_embeddings=True, batch_size=8, show_progress_bar=False)
        return {"score": round(float(np.dot(vectors[0], vectors[1])*100),1), "available": True}
    except Exception: return {"score": None, "available": False}

def compare_analysis(resume: str, jd: str, resume_sections: dict, jd_info: dict, resume_skills: list[str], jd_skills: list[str]) -> dict:
    """Compare role requirements with relevant resume evidence, not whole PDFs.

    Transformer models truncate long documents. Keeping comparisons focused on
    sections and extracted role signals makes the score faster and less biased
    toward text that happens to occur at the start of a PDF.
    """
    requirements = " ".join([jd_info.get("title", ""), *jd_info.get("responsibilities", []), *jd_info.get("required_skills", []), *jd_info.get("preferred_skills", [])])
    summary = resume_sections.get("summary", resume_sections.get("objective", ""))
    experience = " ".join([resume_sections.get("experience", ""), resume_sections.get("internship", "")]).strip()
    pairs = [("summary", summary, requirements, .25), ("experience", experience, requirements, .45), ("skills", " ".join(resume_skills), " ".join(jd_skills), .30)]
    available = [(name, left, right, weight) for name,left,right,weight in pairs if left.strip() and right.strip()]
    if not available: return {"score": None, "available": False, "details": {}, "message": "There was not enough text to compare semantically."}
    try:
        flat = [t for _,left,right,_ in available for t in (left,right)]
        vectors = _model().encode(flat, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
        details = {name: round(float(np.dot(vectors[i*2], vectors[i*2+1])*100),1) for i,(name,_,_,_) in enumerate(available)}
        weight_sum = sum(weight for _,_,_,weight in available)
        weighted = sum(details[name]*weight for name,_,_,weight in available)/weight_sum
        return {"score": round(weighted,1), "available": True, "details": details, "message": None}
    except ModuleNotFoundError as exc:
        missing=exc.name or "a model dependency"
        if missing.split(".")[0] == "sentence_transformers": message="sentence-transformers is missing. Install it with this project's Python environment to enable semantic scoring."
        elif missing.split(".")[0] == "torch": message="PyTorch is missing. Install the project's semantic dependencies to enable embeddings."
        else: message="A semantic-model dependency is missing. Reinstall the project's requirements."
        return {"score": None, "available": False, "details": {}, "message": message}
    except Exception as exc:
        message="The all-MiniLM-L6-v2 model could not load. Check internet access for its first download, then verify that PyTorch has enough memory to run it."
        if "connect" in str(exc).lower() or "huggingface" in str(exc).lower(): message="The model could not be downloaded from Hugging Face. Check internet access; after its first download it can load from the local cache."
        return {"score": None, "available": False, "details": {}, "message": message}
