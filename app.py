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
    if "phd" in text_lower or "doctorate" in text_lower:
        return "PhD"
    elif "master" in text_lower or "ms" in text_lower or "m.sc" in text_lower:
        return "Master's"
    elif "bachelor" in text_lower or "bs" in text_lower or "b.tech" in text_lower:
        return "Bachelor's"
    else:
        return "Other"

def count_certifications(resume_text):
    return len(re.findall(r"certified|certification|certificate", resume_text, re.IGNORECASE))

def count_projects(resume_text):
    return len(re.findall(r"project", resume_text, re.IGNORECASE))

education_map = {
    "Other": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}

def extract_features_for_score(text, extracted_skills):
    """
    Extract features based on updated logic:
      1. Experience (years)
      2. Skills count
      3. Education level (encoded)
      4. Certifications count
      5. Projects count
    """
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
    """
    Load the XGBoost model from 'resume_score_model11.pkl', predict a raw score,
    and then scale it to a 1-10 range.
    """
    try:
        with open("resume_score_model11.pkl", "rb") as score_model_file:
            score_model = pickle.load(score_model_file)
        print("[DEBUG] XGBoost scoring model loaded.")
    except Exception as e:
        print(f"[ERROR] Loading scoring model: {e}")
        return None
    
    features = extract_features_for_score(text, extracted_skills)
    try:
        raw_score = score_model.predict([features])[0]  # Raw predicted score (assumed 0-40)
        # Scale the score to a 1-10 range: raw_score 0 -> 1, raw_score 40 -> 10
        scaled_score = (raw_score / 40) * 9 + 1
        scaled_score = max(1, min(10, scaled_score))
        scaled_score = round(scaled_score, 3)
        print(f"[DEBUG] Predicted Score (scaled 1-10): {scaled_score}")
        return scaled_score
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None

def get_resume_category(score):
    """
    Categorize the resume based on the scaled score (1 to 10):
      - Good: score >= 8
      - Moderate: 4 <= score < 8
      - Bad: score < 4
    """
    if score >= 8:
        return "Good"
    elif score >= 4:
        return "Moderate"
    else:
        return "Bad"

# ---------------------- NLP & Resume Setup ---------------------- #

nlp = spacy.load("en_core_web_sm")

common_skills = {
    skill.lower()
    for skill in {
        "Python", "Java", "C++", "C", "JavaScript", "HTML", "CSS",
        "TypeScript", "Swift", "Kotlin", "Go", "Ruby", "PHP", "R", "MATLAB",
        "Perl", "Rust", "Dart", "Scala", "Shell Scripting", "React", "Angular",
        "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Express.js",
        "Laravel", "Bootstrap", "TensorFlow", "PyTorch", "Keras",
        "Scikit-learn", "NLTK", "Pandas", "NumPy", "SQL", "MySQL",
        "PostgreSQL", "MongoDB", "Firebase", "Cassandra", "Oracle", "Redis",
        "MariaDB", "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
        "Terraform", "CI/CD", "Jenkins", "Git", "GitHub", "Cybersecurity",
        "Penetration Testing", "Ubuntu", "Ethical Hacking", "Firewalls",
        "Cryptography", "IDS", "Network Security", "Machine Learning",
        "Deep Learning", "Numpy", "Pandas", "Matplotlib", "Computer Vision",
        "NLP", "Big Data", "Hadoop", "Spark", "Data Analytics", "Power BI",
        "Tableau", "Data Visualization", "Reinforcement Learning",
        "Advanced DSA", "DSA", "Data Structures and Algorithm", "DevOps", "ML",
        "DL", "Image Processing", "JIRA", "Postman", "Excel", "Leadership",
        "Problem-Solving", "Communication", "Time Management", "Adaptability",
        "Teamwork", "Presentation Skills", "Critical Thinking",
        "Decision Making", "Public Speaking", "Project Management"
    }
}

abbreviation_map = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ds": "data science",
    "js": "javascript",
    "html": "hypertext markup language",
    "css": "cascading style sheets",
    "sql": "structured query language",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "azure": "microsoft azure",
    "dsa": "data structure algorithm",
    "mysql": "my structured query language"
}

# ---------------------- Database Connection ---------------------- #

def get_db_connection(db_name="resume_screening_db"):
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="amaan@khan704093",
        database=db_name,
        auth_plugin="mysql_native_password"
    )

# ---------------------- Resume Processing Functions ---------------------- #

def extract_text_from_file(file):
    """
    Extract text from PDF or DOCX.
    If the file is scanned or no text is extracted, return an empty string.
    """
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
    if not text.strip():
        print("[DEBUG] No text extracted from file.")
    else:
        print(f"[DEBUG] Extracted text length: {len(text)} characters.")
    return text.strip()

def extract_skills(text):
    extracted_skills = set()
    if not text:
        return []
    doc = nlp(text)
    for token in doc:
        if token.text.lower() in common_skills:
            extracted_skills.add(token.text.lower())
    print(f"[DEBUG] Extracted skills: {extracted_skills}")
    return list(extracted_skills)

def extract_name(text):
    lines = text.split('\n')
    name = lines[0].strip() if lines else None
    print(f"[DEBUG] Extracted name: {name}")
    return name

def load_model_and_vectorizer():
    """
    Load your pre-trained classification model and TF-IDF vectorizer.
    Returns (model, vectorizer) if successful.
    """
    try:
        with open("model.pkl", "rb") as model_file:
            rf = pickle.load(model_file)
        with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)
        print("[DEBUG] Classification model and vectorizer loaded successfully.")
        return rf, tfidf
    except Exception as e:
        print(f"[ERROR] Failed to load model/vectorizer: {e}")
        return None, None

# ---------------------- Main Resume Processing ---------------------- #

def process_resume(file):
    """
    Extract text, name, and skills from the resume.
    Then run the classification model to predict the job role and
    the scoring model to predict a resume score.
    Returns:
      (error_message, predicted_job, extracted_skills, user_name, resume_country, resume_score, resume_category)
    """
    rf, tfidf = load_model_and_vectorizer()
    if not rf or not tfidf:
        error_msg = "[ERROR] ML model is missing!"
        print(error_msg)
        return error_msg, None, [], None, "india", None, None
    
    text = extract_text_from_file(file)
    if not text:
        error_msg = "[ERROR] No readable text found in resume!"
        print(error_msg)
        return error_msg, None, [], None, "india", None, None
    
    user_name = extract_name(text)
    extracted_skills = extract_skills(text)
    resume_country = "india"  # Default to India
    
    try:
        text_vectorized = tfidf.transform([text])
        predicted_job = rf.predict(text_vectorized)[0]
        print(f"[DEBUG] Predicted job role: {predicted_job}")
    except Exception as e:
        error_msg = f"[ERROR] Prediction failed: {e}"
        print(error_msg)
        return error_msg, None, extracted_skills, user_name, resume_country, None, None
    
    # Predict resume score using the updated scoring model
    resume_score = predict_resume_score(text, extracted_skills)
    if resume_score is None:
        print("[DEBUG] Resume score prediction returned None.")
    resume_category = get_resume_category(resume_score) if resume_score is not None else None
    print(f"[DEBUG] Resume category: {resume_category}")
    
    return None, predicted_job, extracted_skills, user_name, resume_country, resume_score, resume_category

def compare_skills(predicted_job, extracted_skills, user_name):
    """
    Compares extracted skills with the required skills for predicted_job.
    Inserts missing skills into recommendskills table if any are missing.
    """
    if not predicted_job:
        return []
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT skills FROM jobrolesskills WHERE job_role = %s", (predicted_job,))
        job_data = cursor.fetchone()
        
        if not job_data:
            return []
        
        required_skills = set(job_data["skills"].lower().split(", "))
        extracted_skills_set = set(skill.lower() for skill in extracted_skills)
        missing_skills = required_skills - extracted_skills_set
        
        if missing_skills:
            cursor.execute(
                "INSERT INTO recommendskills (name, job_role, missing_skills) VALUES (%s, %s, %s)",
                (user_name, predicted_job, ", ".join(missing_skills))
            )
            conn.commit()
        
        cursor.close()
        conn.close()
        print(f"[DEBUG] Missing skills: {missing_skills}")
        return list(missing_skills)
    except Exception as e:
        print(f"[ERROR] Skill comparison failed: {e}")
        return []

# ---------------------- Job Listings via API ---------------------- #

def fetch_job_listings_from_api(query="developer", country="india", page=1, job_type=None, remote=None, date_posted=None, salary_range=None, sort_by=None):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-key": "f126e81261msha4b09d552d563fdp193eacjsndb819abae05f",  # Go to README.md file
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }
    params = {
        "query": query,
        "country": country,
        "page": page
    }
    if job_type:
        params["job_type"] = job_type
    if remote is not None:
        params["remote"] = remote
    if date_posted:
        params["date_posted"] = date_posted
    if salary_range:
        params["salary_range"] = salary_range
    if sort_by:
        params["sort_by"] = sort_by

    response = requests.get(url, headers=headers, params=params)
    return response.text

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
    predicted_job = None
    error_message = None
    extracted_skills = []
    missing_skills = []
    user_name = ""
    job_list = []
    resume_country = "india"  # Fallback if extraction fails
    resume_score = None
    resume_category = None

    if request.method == "POST":
        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                # Process the resume
                error_message, predicted_job, extracted_skills, user_name, resume_country, resume_score, resume_category = process_resume(file)
                
                if not error_message:
                    # Compare skills only if we got a valid predicted job
                    missing_skills = compare_skills(predicted_job, extracted_skills, user_name)
                    
                    # Insert resume details into DB along with the predicted score
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO resumes (name, skills, score) VALUES (%s, %s, %s)",
                          (user_name or "Unknown", ", ".join(extracted_skills), float(resume_score)))

                        conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception as db_error:
                        error_message = f"[ERROR] Database error: {db_error}"
                        print(error_message)
        
        # Always try to fetch job listings after processing
        if not error_message:
            job_list = []
            if extracted_skills:
                for skill in extracted_skills:
                    job_listings_json = fetch_job_listings_from_api(query=skill, country=resume_country)
                    try:
                        job_listings_data = json.loads(job_listings_json)
                        if isinstance(job_listings_data, dict) and "data" in job_listings_data:
                            job_list.extend(job_listings_data["data"])
                    except Exception as e:
                        print(f"[ERROR] Failed to fetch or parse job listings for {skill}: {e}")
            elif predicted_job:
                job_listings_json = fetch_job_listings_from_api(query=predicted_job, country=resume_country)
                try:
                    job_listings_data = json.loads(job_listings_json)
                    if isinstance(job_listings_data, dict) and "data" in job_listings_data:
                        job_list = job_listings_data["data"]
                except Exception as e:
                    print(f"[ERROR] Failed to fetch or parse job listings for {predicted_job}: {e}")
            else:
                job_listings_json = fetch_job_listings_from_api(query="developer", country=resume_country)
                try:
                    job_listings_data = json.loads(job_listings_json)
                    if isinstance(job_listings_data, dict) and "data" in job_listings_data:
                        job_list = job_listings_data["data"]
                except Exception as e:
                    print(f"[ERROR] Failed to fetch or parse job listings for 'developer': {e}")

    return render_template("index.html",
                           user_name=user_name or "",
                           predicted_job=predicted_job or "",
                           resume_score=resume_score,
                           resume_category=resume_category,
                           error_message=error_message or "",
                           extracted_skills=extracted_skills,
                           missing_skills=missing_skills,
                           job_list=job_list)

if __name__ == "__main__":
    app.run(debug=True)
