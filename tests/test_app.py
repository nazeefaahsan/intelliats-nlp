import io
import pymupdf as fitz
import sqlite3
import app as webapp

def pdf_bytes(text):
    doc=fitz.open(); page=doc.new_page(); page.insert_text((72,72),text)
    data=doc.tobytes(); doc.close(); return data

def test_home_and_missing_upload():
    webapp.app.config["TESTING"]=True
    client=webapp.app.test_client()
    assert client.get("/").status_code==200
    response=client.post("/analyze",data={})
    assert response.status_code==302

def test_successful_analysis_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp,"DATABASE_PATH",tmp_path/"test.db")
    webapp.ANALYSIS_CACHE.clear()
    client=webapp.app.test_client()
    response=client.post("/analyze",data={"resume_pdf":(io.BytesIO(pdf_bytes("Skills\nPython, Pandas, machine learning\nProjects\nBuilt a model")),"resume.pdf","application/pdf"),"jd_pdf":(io.BytesIO(pdf_bytes("Machine Learning Intern\nRequired: Python and Pandas. Develop machine learning models.")),"job.pdf","application/pdf")},content_type="multipart/form-data",follow_redirects=True)
    assert response.status_code==200
    assert b"Your compatibility report" in response.data
    assert client.get("/history").status_code==200
    with sqlite3.connect(tmp_path/"test.db") as con:
        columns={row[1] for row in con.execute("PRAGMA table_info(analyses)")}
        assert columns=={"id","timestamp","job_title","ats_score","semantic_score","tfidf_score","skill_score"}
        analysis_id=con.execute("SELECT id FROM analyses").fetchone()[0]
        con.execute("UPDATE analyses SET semantic_score=NULL WHERE id=?",(analysis_id,))
    webapp.ANALYSIS_CACHE.pop(analysis_id,None)
    old_report=client.get(f"/results/{analysis_id}")
    assert old_report.status_code==200
    assert b"predates semantic-model availability" in old_report.data
