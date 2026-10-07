"""Extract requirements and likely role title from a job description."""
import re
from nlp.skill_extractor import taxonomy
REQUIRED = ("required", "must have", "mandatory", "essential", "proficient in", "strong knowledge", "should have")
PREFERRED = ("preferred", "nice to have", "added advantage", "bonus", "familiarity with", "a plus")
SOFT_TERMS = ("communication", "teamwork", "collaboration", "problem solving", "critical thinking", "adaptability", "leadership", "attention to detail", "time management")
def analyze_jd(text: str, skills: list[str]) -> dict:
    lines = [x.strip(" •\t-") for x in (text or "").splitlines() if x.strip()]
    role_pattern=r"\b(intern(?:ship)?|engineer|analyst|designer|developer|manager|scientist|specialist|consultant|coordinator|architect|associate)\b"
    title=next((x for x in lines[:15] if re.search(role_pattern,x,re.I) and len(x)<=160),None)
    if not title:
        generic={"job description","role description","responsibilities","about the job","position overview"}
        title=next((x for x in lines[:6] if x.lower().strip(" :") not in generic and len(x.split())<=10 and len(x)<=120),"Role not specified")
    lower = (text or "").lower()
    aliases = {"ml": "machine learning", "nlp": "natural language processing", "sklearn": "scikit-learn", "nodejs": "node.js", "reactjs": "react.js"}
    def nearby_phrases(skill):
        forms = [skill.lower()] + [alias for alias, canonical in aliases.items() if canonical == skill.lower()]
        contexts = []
        for form in forms:
            for match in re.finditer(r"(?<![\w])" + re.escape(form) + r"(?![\w])", lower):
                contexts.append(lower[max(0, match.start()-70):match.end()+70])
        return contexts
    req = [s for s in skills if any(p in context for context in nearby_phrases(s) for p in REQUIRED)]
    pref = [s for s in skills if any(p in context for context in nearby_phrases(s) for p in PREFERRED)]
    categories = taxonomy()
    return {"title": title[:120], "required_skills": req, "preferred_skills": pref, "technical_requirements": skills, "tools": [s for s in skills if s in categories.get("tools", [])], "soft_skills": [term for term in SOFT_TERMS if term in lower], "responsibilities": [x for x in lines if re.search(r"\b(develop|design|build|analy[sz]e|create|support|manage|conduct|evaluate|implement)\b",x,re.I)][:12], "education": [x for x in lines if re.search(r"\b(bachelor|degree|master|education|graduate)\b",x,re.I)][:5], "experience": [x for x in lines if re.search(r"\b(years?|experience)\b",x,re.I)][:5]}
