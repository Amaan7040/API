from flask import Flask, jsonify, render_template, request
import http.client
import json
import urllib.parse
import os
import pickle
import re
import docx
import pdfplumber
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.matcher import PhraseMatcher
import mysql.connector
import requests
import numpy as np

# ---------------------- Resume Scoring Features ---------------------- #

def extract_experience_safe(resume_text):
    experience_patterns = [
        r'(\d+)\s*[\+]?[\s-]*(?:years?|yrs?)\s*(?:of)?\s*experience',
        r'experience\s*(?:of)?\s*(\d+)\s*[\+]?[\s-]*(?:years?|yrs?)'
    ]
    for pattern in experience_patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return 0
    return 0

def extract_education(resume_text):
    text_lower = resume_text.lower()
    # Check diploma before bachelor so that "diploma" is not overshadowed by bachelor keywords
    if "phd" in text_lower or "doctorate" in text_lower:
        return "PhD"
    elif "master" in text_lower or "ms" in text_lower or "m.sc" in text_lower:
        return "Master's"
    elif "bachelor" in text_lower or "bs" in text_lower or "b.tech" in text_lower:
        return "Bachelor's"
    elif "diploma" in text_lower:
        return "Diploma"
    else:
        return "Other"

def count_certifications(resume_text):
    return len(re.findall(r"certified|certification|certificate", resume_text, re.IGNORECASE))

def count_projects(resume_text):
    return len(re.findall(r"project", resume_text, re.IGNORECASE))

# Updated education mapping:
# Other: 0, Diploma: 1, Bachelor's: 2, Master's: 3, PhD: 4
education_map = {
    "Other": 0,
    "Diploma": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "PhD": 4
}

def extract_features_for_score(text, extracted_skills):
    experience = extract_experience_safe(text)
    skills_count = len(extracted_skills)
    education_level = extract_education(text)
    education_encoded = education_map.get(education_level, 0)
    certifications_count = count_certifications(text)
    projects_count = count_projects(text)
    
    features = [experience, skills_count, education_encoded, certifications_count, projects_count]
    print(f"[DEBUG] Score Features: {features}")
    return features

def predict_resume_score(text, extracted_skills):
    try:
        with open("resume_score_model11.pkl", "rb") as score_model_file:
            score_model = pickle.load(score_model_file)
        print("[DEBUG] XGBoost scoring model loaded.")
    except Exception as e:
        print(f"[ERROR] Loading scoring model: {e}")
        return None
    
    features = extract_features_for_score(text, extracted_skills)
    try:
        raw_score = score_model.predict([features])[0]
        scaled_score = (raw_score / 40) * 9 + 1
        scaled_score = max(1, min(10, scaled_score))
        scaled_score = round(scaled_score, 3)
        print(f"[DEBUG] Predicted Score (scaled 1-10): {scaled_score}")
        return scaled_score
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None

def get_resume_category(score):
    if score >= 8:
        return "Good"
    elif score >= 4:
        return "Moderate"
    else:
        return "Bad"

# ---------------------- NLP & Resume Setup ---------------------- #

nlp = spacy.load("en_core_web_sm")

# Technical skills set
technical_skills = {
    "python", "java", "c++", "c", "javascript", "html", "css",
    "typescript", "swift", "kotlin", "go", "ruby", "php", "r", "matlab",
    "perl", "rust", "dart", "scala", "shell scripting", "react", "angular",
    "vue.js", "node.js", "django", "flask", "spring boot", "express.js",
    "laravel", "bootstrap", "tensorflow", "pytorch", "keras",
    "scikit learn", "nltk", "pandas", "numpy", "sql", "mysql",
    "postgresql", "mongodb", "firebase", "cassandra", "oracle", "redis",
    "mariadb", "aws", "azure", "google cloud", "docker", "kubernetes",
    "terraform", "ci/cd", "jenkins", "git", "github", "cybersecurity",
    "penetration testing", "ubuntu", "ethical hacking", "firewalls",
    "cryptography", "ids", "network security", "machine learning",
    "deep learning", "matplotlib", "computer vision",
    "natural language processing", "big data", "hadoop", "spark", 
    "data analytics", "power bi", "tableau", "data visualization",
    "reinforcement learning", "advanced dsa", "data structures and algorithm",
    "devops", "image processing", "jira", "postman", "excel", "data preprocessing",
    "matplotlib", "seaborn", "api integration"
}

# Non-technical (soft) skills
non_technical_skills = {
    "leadership", "problem-solving", "communication", "time management",
    "adaptability", "teamwork", "presentation skills", "critical thinking",
    "decision making", "public speaking", "project management",
    "customer service", "organization", "strategic planning",
    "relationship management", "creativity", "negotiation", "collaboration"
}

# Merge both sets for enhanced extraction
common_skills = technical_skills.union(non_technical_skills)

# Abbreviation map for common abbreviations
abbreviation_map = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ds": "data science",
    "js": "javascript",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "azure": "microsoft azure",
    "dsa": "data structure algorithm"
}

# ----------------------- Skill Extraction ---------------------- #

def expand_abbreviations(text, abbreviation_map):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(abbr) for abbr in abbreviation_map.keys()) + r')\b', re.IGNORECASE)
    return pattern.sub(lambda x: abbreviation_map[x.group().lower()], text)

def extract_skills(text):
    expanded_text = expand_abbreviations(text, abbreviation_map)
    doc = nlp(expanded_text.lower())
    
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    skills_patterns = [nlp.make_doc(skill) for skill in common_skills]
    matcher.add("SKILLS", None, *skills_patterns)
    
    matches = matcher(doc)
    extracted_skills = set()
    
    for match_id, start, end in matches:
        span = doc[start:end]
        skill = span.text.strip()
        if skill:
            extracted_skills.add(skill)
    
    for token in doc:
        if token.text in common_skills and len(token.text) > 2:
            extracted_skills.add(token.text)
    
    formatted_skills = set()
    for skill in extracted_skills:
        clean_skill = re.sub(r'[^\w\s-]', '', skill).strip()
        if clean_skill:
            formatted_skills.add(clean_skill.title())
    
    print(f"[DEBUG] Enhanced skills extraction: {formatted_skills}")
    return sorted(formatted_skills, key=lambda x: (-len(x), x))

# ---------------------- Database Connection ---------------------- #

def get_db_connection(db_name="resume_screening_db"):
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="", # your db password
        database=db_name,
        auth_plugin="mysql_native_password"
    )

# ---------------------- Resume Processing Functions ---------------------- #

def extract_text_from_file(file):
    text = ""
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_name(text):
    lines = text.split('\n')
    return lines[0].strip() if lines else None

def load_model_and_vectorizer():
    try:
        with open("combined_job_predict_model.pkl", "rb") as model_file:
            rf = pickle.load(model_file)
        with open("combined_tfidf_vectorizer.pkl", "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)
        return rf, tfidf
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None, None

# ---------------------- Main Resume Processing ---------------------- #

def process_resume(file):
    rf, tfidf = load_model_and_vectorizer()
    if not rf or not tfidf:
        return "ML model missing!", None, [], None, "india", None, None
    
    text = extract_text_from_file(file)
    if not text:
        return "No readable text!", None, [], None, "india", None, None
    
    user_name = extract_name(text)
    extracted_skills = extract_skills(text)
    resume_country = "india"
    
    try:
        text_vectorized = tfidf.transform([text])
        predicted_job = rf.predict(text_vectorized)[0]
    except Exception as e:
        return f"Prediction failed: {e}", None, extracted_skills, user_name, resume_country, None, None
    
    resume_score = predict_resume_score(text, extracted_skills)
    resume_category = get_resume_category(resume_score) if resume_score else None
    
    return None, predicted_job, extracted_skills, user_name, resume_country, resume_score, resume_category

def compare_skills(predicted_job, extracted_skills, user_name):
    if not predicted_job:
        return []
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT skills FROM jobrolesskills WHERE job_role = %s", (predicted_job,))
        job_data = cursor.fetchone()
        
        if job_data:
            required_skills = set(job_data["skills"].lower().split(", "))
            extracted_skills_set = set(s.lower() for s in extracted_skills)
            missing_skills = required_skills - extracted_skills_set
            
            if missing_skills:
                cursor.execute(
                    "INSERT INTO recommendskills (name, job_role, missing_skills) VALUES (%s, %s, %s)",
                    (user_name, predicted_job, ", ".join(missing_skills))
                )
                conn.commit()
        cursor.close()
        conn.close()
        return list(missing_skills)
    except Exception as e:
        print(f"[ERROR] Skill comparison: {e}")
        return []

# ---------------------- Flask Application Setup ---------------------- #

app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/api/job-details", methods=["GET"])
def job_details_api():
    try:
        job_data = fetch_job_listings_from_api()
        return jsonify({"success": True, "data": job_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_job = error_message = None
    extracted_skills = []
    missing_skills = []
    user_name = ""
    job_list = []
    resume_country = "india"
    resume_score = resume_category = None

    if request.method == "POST":
        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                error_message, predicted_job, extracted_skills, user_name, resume_country, resume_score, resume_category = process_resume(file)
                
                if not error_message:
                    missing_skills = compare_skills(predicted_job, extracted_skills, user_name)
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO resumes (name, skills, score) VALUES (%s, %s, %s)",
                                       (user_name or "Unknown", ", ".join(extracted_skills), float(resume_score)))
                        conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception as e:
                        error_message = f"Database error: {e}"

        if not error_message:
            job_list = []
            search_terms = extracted_skills if extracted_skills else ([predicted_job] if predicted_job else ["developer"])
            for term in search_terms[:3]:
                try:
                    job_listings_json = fetch_job_listings_from_api(query=term, country=resume_country)
                    job_listings_data = json.loads(job_listings_json)
                    if isinstance(job_listings_data, dict) and "data" in job_listings_data:
                        job_list.extend(job_listings_data["data"][:5])
                except Exception as e:
                    print(f"Job search error for {term}: {e}")

    return render_template("index.html",
                           user_name=user_name or "",
                           predicted_job=predicted_job or "",
                           resume_score=resume_score,
                           resume_category=resume_category,
                           error_message=error_message or "",
                           extracted_skills=extracted_skills,
                           missing_skills=missing_skills,
                           job_list=job_list)

def fetch_job_listings_from_api(query="developer", country="india", page=1, job_type=None):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-key": "", # your API Key 
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }
    params = {"query": f"{query} in {country}", "page": page, "num_pages": 1}
    if job_type:
        params["job_type"] = job_type
    response = requests.get(url, headers=headers, params=params)
    return response.text

if __name__ == "__main__":
    app.run(debug=True)
