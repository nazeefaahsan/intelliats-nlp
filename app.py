"""IntelliATS Flask application."""
from datetime import datetime
from pathlib import Path
import re
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort
from werkzeug.utils import secure_filename
from config import MAX_UPLOAD_BYTES, SECRET_KEY, DATABASE_PATH, SEMANTIC_ENABLED
from utils.pdf_parser import extract_text_from_pdf, PDFError
from utils.validators import is_pdf
from nlp.preprocessing import preprocess
from nlp.skill_extractor import extract_skills, flatten
from nlp.tfidf_matcher import compare as tfidf_compare
from nlp.keyword_extractor import extract_keywords
from nlp.semantic_matcher import compare_analysis as semantic_compare
from nlp.section_parser import parse_sections
from nlp.jd_analyzer import analyze_jd
from scoring.ats_score import score_analysis
from scoring.recommendations import recommendations

app = Flask(__name__); app.secret_key = SECRET_KEY; app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES * 2 + 64 * 1024
ANALYSIS_CACHE = {}
def db():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con=sqlite3.connect(DATABASE_PATH); con.row_factory=sqlite3.Row
    con.execute("CREATE TABLE IF NOT EXISTS analyses(id INTEGER PRIMARY KEY, timestamp TEXT, job_title TEXT, ats_score REAL, semantic_score REAL, tfidf_score REAL, skill_score REAL)")
    return con
@app.errorhandler(413)
def too_large(_): return render_template("error.html", message="Each PDF must be 10 MB or smaller."), 413
@app.route("/")
def home(): return render_template("index.html")
@app.post("/analyze")
def analyze():
    resume=request.files.get("resume_pdf"); jd=request.files.get("jd_pdf")
    if not resume or not resume.filename: flash("Choose a resume PDF to continue.", "error"); return redirect(url_for("home"))
    if not jd or not jd.filename: flash("Choose a job description PDF to continue.", "error"); return redirect(url_for("home"))
    for upload in (resume,jd):
        safe=secure_filename(upload.filename)
        if not is_pdf(safe, upload.mimetype): flash("Upload PDF files only.", "error"); return redirect(url_for("home"))
        upload.stream.seek(0, 2); size=upload.stream.tell(); upload.stream.seek(0)
        if size > MAX_UPLOAD_BYTES: flash("Each PDF must be 10 MB or smaller.", "error"); return redirect(url_for("home"))
    try:
        resume_text=extract_text_from_pdf(resume.stream); jd_text=extract_text_from_pdf(jd.stream)
    except PDFError as exc: flash(str(exc), "error"); return redirect(url_for("home"))
    rp=preprocess(resume_text); jp=preprocess(jd_text)
    resume_skills=extract_skills(resume_text); jd_skills=extract_skills(jd_text)
    rskills=flatten(resume_skills); jskills=flatten(jd_skills)
    jd_info=analyze_jd(jd_text,jskills)
    matched=sorted(set(rskills)&set(jskills), key=str.lower); missing=sorted(set(jskills)-set(rskills), key=str.lower)
    required=set(jd_info["required_skills"])
    required_score=(len(required & set(rskills))/len(required)*100) if required else None
    overall_skill_score=(len(matched)/len(jskills)*100) if jskills else 0.0
    skill_score=round(.70*required_score+.30*overall_skill_score if required_score is not None else overall_skill_score,1)
    sections=parse_sections(resume_text)
    tfidf=tfidf_compare(rp.normalized_text,jp.normalized_text,sections,jd_info)
    semantic=semantic_compare(resume_text,jd_text,sections,jd_info,rskills,jskills) if SEMANTIC_ENABLED else {"score":None,"available":False,"details":{},"message":"Semantic embeddings are disabled for fast analysis. Set INTELLIATS_SEMANTIC=1 to enable them."}
    resume_lower=resume_text.lower()
    keywords=extract_keywords(jd_text,tfidf["top_terms"],jskills,jd_info["required_skills"]); keyword_match=[x for x in keywords if re.search(r"(?<![\w])"+re.escape(x.lower())+r"(?![\w])",resume_lower)]
    keyword_coverage=(len(keyword_match)/len(keywords)) if keywords else 0.0
    required_coverage=(len(set(jd_info["required_skills"])&set(rskills))/len(jd_info["required_skills"])) if jd_info["required_skills"] else keyword_coverage
    keyword_score=round(100*(.7*keyword_coverage+.3*required_coverage),1)
    section_weights={"summary":.10,"objective":.05,"skills":.25,"experience":.25,"internship":.15,"projects":.20}
    applicable={key:weight for key,weight in section_weights.items() if key in sections}
    structure_score=round(100*sum(applicable.values()),1)
    required_matched=sorted(required & set(rskills), key=str.lower)
    required_skill_coverage=round(100*len(required_matched)/len(required),1) if required else None
    keyword_gaps=[k for k in keywords if not re.search(r"(?<![\w])"+re.escape(k.lower())+r"(?![\w])",resume_lower)]
    skill_categories=[]
    for category,category_skills in jd_skills.items():
        category_matches=set(category_skills)&set(rskills)
        total=len(set(category_skills))
        if total:
            skill_categories.append({"name":category.replace("_"," ").title(),"matched":len(category_matches),"total":total,"coverage":round(100*len(category_matches)/total,1)})
    ats,weights=score_analysis(semantic["score"],tfidf["score"],skill_score,keyword_score,structure_score)
    payload={"score":ats,"components":{"semantic":semantic["score"],"tfidf":tfidf["score"],"skills":skill_score,"keywords":keyword_score,"structure":structure_score},"semantic_details":semantic.get("details",{}),"semantic_message":semantic.get("message"),"weights":weights,"semantic_available":semantic["available"],"resume_skills":resume_skills,"jd_skills":jd_skills,"matched":matched,"missing":missing,"required_matched":required_matched,"required_skill_count":len(required),"required_skill_coverage":required_skill_coverage,"keyword_coverage":round(keyword_coverage*100,1),"skill_categories":skill_categories,"keywords":keywords,"keyword_match":keyword_match,"keyword_gaps":keyword_gaps,"sections":sections,"jd":jd_info,"tfidf_document_score":tfidf["document_score"],"tfidf_context_score":tfidf["context_score"],"relevant_evidence":tfidf["evidence"],"tokens":rp.tokens[:100],"lemmas":rp.lemmas[:100],"recommendations":recommendations(missing,sections,keyword_gaps),"resume_name":secure_filename(resume.filename),"jd_name":secure_filename(jd.filename)}
    con=db(); cur=con.execute("INSERT INTO analyses(timestamp,job_title,ats_score,semantic_score,tfidf_score,skill_score) VALUES(?,?,?,?,?,?)",(datetime.now().isoformat(timespec="minutes"),jd_info["title"],ats,semantic["score"],tfidf["score"],skill_score)); row_id=cur.lastrowid; con.commit(); con.close()
    ANALYSIS_CACHE[row_id]=payload
    return redirect(url_for("results", analysis_id=row_id))
@app.route("/results/<int:analysis_id>")
def results(analysis_id):
    con=db(); row=con.execute("SELECT * FROM analyses WHERE id=?",(analysis_id,)).fetchone(); con.close()
    if not row: abort(404)
    data=ANALYSIS_CACHE.get(analysis_id)
    if data is None:
        data={"score":row["ats_score"],"components":{"semantic":row["semantic_score"],"tfidf":row["tfidf_score"],"skills":row["skill_score"],"keywords":0,"structure":0},"weights":{},"semantic_available":row["semantic_score"] is not None,"semantic_message":"This saved report predates semantic-model availability. Uploaded PDFs are not retained, so upload both files again to calculate a semantic score." if row["semantic_score"] is None else None,"resume_skills":{},"jd_skills":{},"matched":[],"missing":[],"keywords":[],"keyword_match":[],"sections":{},"jd":{"title":row["job_title"],"preferred_skills":[],"responsibilities":[],"education":[],"experience":[]},"tokens":[],"lemmas":[],"recommendations":["This report's detailed breakdown is no longer in the temporary session. Run the analysis again to view the full report."],"resume_name":"","jd_name":""}
    return render_template("results.html", data=data)
@app.route("/history")
def history():
    con=db(); rows=con.execute("SELECT id,timestamp,job_title,ats_score FROM analyses ORDER BY id DESC LIMIT 50").fetchall(); con.close()
    return render_template("history.html", rows=rows)
@app.route("/api/analysis/<int:analysis_id>")
def api_analysis(analysis_id):
    con=db(); row=con.execute("SELECT * FROM analyses WHERE id=?",(analysis_id,)).fetchone(); con.close()
    if not row: abort(404)
    data=ANALYSIS_CACHE.get(analysis_id)
    if data is None: return jsonify({"id":row["id"],"timestamp":row["timestamp"],"job_title":row["job_title"],"score":row["ats_score"],"semantic_score":row["semantic_score"],"tfidf_score":row["tfidf_score"],"skill_score":row["skill_score"],"detail_available":False})
    return jsonify(data)
@app.errorhandler(500)
def server_error(_): return render_template("error.html",message="We couldn't complete this analysis. Please check both PDFs and try again."),500
if __name__ == "__main__": app.run(debug=False)
