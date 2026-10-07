"""Inspectable linguistic features using spaCy's POS, parser and NER pipelines."""
from collections import Counter
import re
from nlp.preprocessing import _lemmatizer

def _syllables(word: str) -> int:
    parts=re.findall(r"[aeiouy]+",word.lower().strip(".,!?;:"))
    count=len(parts)
    if word.lower().endswith("e") and count>1: count-=1
    return max(count,1)

def analyze_linguistics(text: str) -> dict:
    text=(text or "")[:100000]
    model=_lemmatizer()
    if model is None:
        words=re.findall(r"\b[a-zA-Z]{2,}\b",text)
        sentences=[s for s in re.split(r"[.!?]+",text) if s.strip()]
        return {"model":"rule-based fallback","entities":[],"pos_counts":{},"noun_phrases":[],"relations":[],"sentence_count":len(sentences),"lexical_diversity":round(len(set(w.lower() for w in words))/max(len(words),1),3),"avg_sentence_words":round(len(words)/max(len(sentences),1),1),"flesch_reading_ease":None}
    doc=model(text)
    words=[t for t in doc if t.is_alpha]
    sentences=list(doc.sents)
    pos=Counter(t.pos_ for t in words if t.pos_)
    entities=[]; seen=set()
    for ent in doc.ents:
        item=(ent.text.strip(),ent.label_)
        if item not in seen: entities.append({"text":item[0],"type":item[1]}); seen.add(item)
    phrase_counts=Counter(" ".join(t.lemma_.lower() for t in chunk if not t.is_stop and t.is_alpha) for chunk in doc.noun_chunks)
    phrase_counts=Counter({phrase:count for phrase,count in phrase_counts.items() if len(phrase.split())>1 and len(phrase)>4})
    relations=[]
    for subject in (t for t in doc if t.dep_ in {"nsubj","nsubjpass"}):
        obj=next((c for c in subject.head.children if c.dep_ in {"dobj","obj","attr","oprd"}),None)
        if obj:
            relations.append({"subject":subject.text,"action":subject.head.lemma_,"object":obj.text})
    syllable_count=sum(_syllables(t.text) for t in words)
    word_count=max(len(words),1); sentence_count=max(len(sentences),1)
    flesch=206.835-1.015*(word_count/sentence_count)-84.6*(syllable_count/word_count)
    return {"model":"spaCy en_core_web_sm","entities":entities[:20],"pos_counts":dict(pos.most_common()),"noun_phrases":[{"text":p,"count":n} for p,n in phrase_counts.most_common(12)],"relations":relations[:10],"sentence_count":len(sentences),"lexical_diversity":round(len({t.lemma_.lower() for t in words})/word_count,3),"avg_sentence_words":round(word_count/sentence_count,1),"flesch_reading_ease":round(max(0,min(100,flesch)),1)}
