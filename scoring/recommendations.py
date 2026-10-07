"""Generate grounded suggestions from observed gaps."""
def recommendations(missing, sections, keyword_gaps):
    result=[]
    if missing: result.append("If you genuinely have these skills, add evidence of them: " + ", ".join(missing[:6]) + ".")
    if not any(k in sections for k in ("projects", "experience", "internship")): result.append("Add a Projects or Experience section with concrete work relevant to this role.")
    if keyword_gaps: result.append("Use relevant terminology naturally where it accurately describes your experience: " + ", ".join(keyword_gaps[:5]) + ".")
    if not result: result.append("Your detected sections and skills align well. Strengthen project bullets with clear outcomes and measurable impact.")
    return result
