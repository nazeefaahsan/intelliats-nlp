"""Simple heading based resume section detection."""
import re
HEADINGS = {"summary": ["summary", "profile", "objective"], "education": ["education", "academic background"], "skills": ["skills", "technical skills"], "experience": ["experience", "work experience", "employment"], "internship": ["internship", "internships"], "projects": ["projects", "project experience"], "certifications": ["certifications", "certificates"], "achievements": ["achievements", "awards"], "publications": ["publications"], "activities": ["activities", "volunteering"]}
def parse_sections(text: str) -> dict:
    found = {key: [] for key in HEADINGS}; active = None
    for line in (text or "").splitlines():
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        heading = next((k for k, names in HEADINGS.items() if normalized in names), None)
        if heading: active = heading
        elif active and line.strip(): found[active].append(line.strip())
    return {k: "\n".join(v) for k,v in found.items() if v}
