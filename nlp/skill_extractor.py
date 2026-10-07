"""Taxonomy-driven skill matching."""
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def taxonomy(): return json.loads((ROOT / "data/skills.json").read_text(encoding="utf-8"))
def synonyms(): return json.loads((ROOT / "data/synonyms.json").read_text(encoding="utf-8"))
def extract_skills(text: str) -> dict:
    source = (text or "").lower(); aliases = synonyms(); result = {}
    for category, skills in taxonomy().items():
        found = []
        for skill in skills:
            forms = [skill] + [k for k,v in aliases.items() if v.lower() == skill.lower()]
            if any(re.search(r"(?<![\w])"+re.escape(f.lower())+r"(?![\w])", source) for f in forms): found.append(skill)
        if found: result[category] = sorted(set(found), key=str.lower)
    return result
def flatten(categorized: dict) -> list[str]: return sorted({s for values in categorized.values() for s in values}, key=str.lower)
