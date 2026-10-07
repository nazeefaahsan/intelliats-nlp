# IntelliATS — NLP-Powered Resume & Job Description Compatibility Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Flask](https://img.shields.io/badge/Flask-3.x-000000)
![NLP](https://img.shields.io/badge/Focus-NLP-6A5ACD)
![License](https://img.shields.io/badge/License-MIT-green)

IntelliATS is an academic resume-analysis app that compares a resume PDF with a job-description PDF. It combines normalized skill matching, TF-IDF, sentence-level relevance, keyword coverage, resume sections, and optional sentence embeddings to produce an explainable compatibility estimate.

It is an educational ATS simulation. It does not reproduce proprietary applicant-tracking systems or predict hiring outcomes.

## Problem Statement

Applicants often need to compare a resume with many role descriptions, while the reasons for a match or mismatch can be hard to see. IntelliATS highlights relevant skills and terms, detected gaps, and the signals that contribute to its score.

## Objective

Provide a local, inspectable NLP workflow and a compatibility score whose components are visible and can be reviewed independently.

## Features

- Upload a resume and a job description as PDFs (up to 10 MB each).
- Extract text in memory and report invalid, empty, or scanned PDFs.
- Normalize technical terms; tokenize, remove stop words, split sentences, and lemmatize with local rules or spaCy.
- Analyze unigrams, bigrams, and trigrams.
- Extract skills with an editable taxonomy and synonym map.
- Identify likely job title, required and preferred skills, responsibilities, education, and experience signals.
- Rank job-description terms and compare documents with TF-IDF and cosine similarity.
- Rank resume sentences by relevance to extracted job requirements and show the supporting evidence.
- Optionally compare relevant resume evidence with sentence embeddings. This is disabled by default to keep analysis fast.
- Show matched and missing skills, recommendations, score components, and detected resume sections.
- Store analysis metadata in local SQLite history; uploaded PDF contents are not stored.
- Provide a JSON analysis endpoint, responsive interface, dark mode, and score charts.

## NLP Pipeline

```text
Resume PDF + job-description PDF
                ↓
       PyMuPDF text extraction
                ↓
Normalization → tokenization → stop-word filtering → lemmatization
                ↓
      N-grams + section detection
                ↓
 Taxonomy skills + JD requirements + keywords
                ↓
 TF-IDF document similarity + requirement-to-sentence relevance
                ↓
 Optional sentence embeddings (opt-in)
                ↓
 Explainable score + gaps + recommendations
```

The project demonstrates tokenization, stop-word removal, lemmatization, n-grams, TF-IDF, cosine similarity, sentence ranking, synonym normalization, phrase-based requirement detection, sentence embeddings (optional), section parsing, keyword extraction, and weighted scoring. The `nlp/linguistic_analyzer.py` module also contains optional spaCy POS, entity, and noun-phrase analysis utilities.

## ATS Scoring

When sentence embeddings are enabled and available, the score is:

```text
ATS = 0.30 × semantic + 0.22 × TF-IDF + 0.28 × skills
    + 0.12 × keywords + 0.08 × structure
```

If embeddings are disabled or unavailable, the remaining signals are renormalized:

```text
ATS = 0.32 × TF-IDF + 0.40 × skills
    + 0.18 × keywords + 0.10 × structure
```

The TF-IDF component blends whole-document cosine similarity (45%) with the mean similarity of the three most relevant resume sentences to extracted job requirements (55%). The report shows the sentences that support this evidence score. The skill component gives 70% weight to detected required skills and 30% to all detected technical skills. If the job description has no detected required skills, it uses the overall technical-skill match. The keyword component combines keyword coverage (70%) and required-skill coverage (30%). Structure is the sum of weights for detected sections: summary 10%, objective 5%, skills 25%, experience 25%, internship 15%, and projects 20%. These are heuristic signals, not a hiring decision.

## Technology Stack

- Python and Flask
- PyMuPDF for PDF text extraction
- scikit-learn and NumPy for TF-IDF and similarity
- spaCy for optional linguistic analysis and model-backed lemmatization
- Sentence Transformers for optional semantic scoring
- Jinja templates, HTML, CSS, JavaScript, and Chart.js
- SQLite for local analysis metadata

## Project Structure

```text
app.py                  Flask routes and analysis workflow
config.py               Application configuration
requirements.txt        Python dependencies
nlp/                    Text preprocessing and NLP analysis modules
scoring/                Compatibility scoring and recommendations
utils/                  PDF validation and text extraction
data/                   Skill taxonomy and synonym JSON files
templates/              Flask/Jinja pages
static/css/              Application styles
static/js/               Browser interactions
tests/                  NLP and Flask route tests
instance/                Runtime SQLite database (ignored by Git)
```

## Installation

```powershell
git clone https://github.com/nazeefaahsan/intelliats-nlp.git
cd intelliats-nlp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

spaCy is optional. To use model-backed lemmatization and linguistic utilities, install spaCy and its English model:

```powershell
pip install spacy
python -m spacy download en_core_web_sm
```

Semantic scoring is optional and disabled by default. Install its package when needed:

```powershell
pip install sentence-transformers
$env:INTELLIATS_SEMANTIC = "1"
```

The first semantic analysis may download `all-MiniLM-L6-v2`. It can take longer than the default lexical and skill-based analysis.

For deployments, set `SECRET_KEY` to a unique random value. A placeholder is provided in `.env.example`; the application does not automatically load `.env` files.

## Running the Application

```powershell
python app.py
```

Open <http://127.0.0.1:5000> in a browser.

## Usage

1. Upload the resume PDF.
2. Upload the job-description PDF.
3. Select **Analyze resume**.
4. Review the compatibility score and its NLP components.
5. Review matched and missing skills, role signals, detected sections, and recommendations.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
pytest -q
```

## Limitations

This is an academic NLP-based ATS simulation, not a replica of commercial ATS software. Scanned PDFs require OCR, which is not included. Skill coverage depends on the editable taxonomy, and requirement classification and section detection use heuristics. Semantic scores require an optional model download. Scores should be treated as guidance rather than an employment prediction.

## Future Scope

- OCR for scanned resumes
- Multilingual resume analysis
- Expanded skill ontology and job-role classification
- More detailed resume-improvement suggestions
- Recruiter-facing dashboard
