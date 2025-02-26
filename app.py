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
    return text.strip()

def extract_skills(text):
    extracted_skills = set()
    if not text:
        return []
    doc = nlp(text)
    for token in doc:
        if token.text.lower() in common_skills:
            extracted_skills.add(token.text.lower())
    return list(extracted_skills)

def extract_name(text):
    lines = text.split('\n')
    return lines[0].strip() if lines else None

def load_model_and_vectorizer():
    """
    Load your pre-trained model and TF-IDF vectorizer.
    Returns (None, None) if loading fails.
    """
    try:
        with open("model.pkl", "rb") as model_file:
            rf = pickle.load(model_file)
        with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)
        return rf, tfidf
    except Exception as e:
        print(f"[ERROR] Failed to load model/vectorizer: {e}")
        return None, None

def process_resume(file):
    """
    Extract text, name, and skills from the resume.
    Then run the model to predict the job role.
    Returns:
      (error_message, predicted_job, extracted_skills, user_name, country)
    """
    rf, tfidf = load_model_and_vectorizer()
    if not rf or not tfidf:
        return "[ERROR] ML model is missing!", None, [], None, "india"
    
    text = extract_text_from_file(file)
    if not text:
        # Possibly scanned or empty PDF, return an error
        return "[ERROR] No readable text found in resume!", None, [], None, "india"
    
    user_name = extract_name(text)
    extracted_skills = extract_skills(text)
    resume_country = "india"  # Default to India
    
    try:
        text_vectorized = tfidf.transform([text])
        predicted_job = rf.predict(text_vectorized)[0]
        return None, predicted_job, extracted_skills, user_name, resume_country
    except Exception as e:
        return f"[ERROR] Prediction failed: {e}", None, extracted_skills, user_name, resume_country

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
        return list(missing_skills)
    except Exception as e:
        print(f"[ERROR] Skill comparison failed: {e}")
        return []

# ---------------------- Job Listings via API ---------------------- #

def fetch_job_listings_from_api(query="developer", country="india", page=1, job_type=None, remote=None, date_posted=None, salary_range=None, sort_by=None):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-key": "4b79c86f33msh4e37937197d933dp148c3ejsne9530e3d6fc5",
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }
    # Build parameters dictionary
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

    if request.method == "POST":
        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                # Process the resume
                error_message, predicted_job, extracted_skills, user_name, resume_country = process_resume(file)
                
                if not error_message:
                    # Compare skills only if we got a valid predicted job
                    missing_skills = compare_skills(predicted_job, extracted_skills, user_name)
                    
                    # Insert resume details into DB
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO resumes (name, skills) VALUES (%s, %s)",
                                       (user_name or "Unknown", ", ".join(extracted_skills)))
                        conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception as db_error:
                        error_message = f"[ERROR] Database error: {db_error}"
        
        # Always try to fetch job listings after processing
        if not error_message:
            job_list = []
            if extracted_skills:
                # Loop through each extracted skill and fetch job listings for each
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
                           error_message=error_message or "",
                           extracted_skills=extracted_skills,
                           missing_skills=missing_skills,
                           job_list=job_list)

if __name__ == "__main__":
    app.run(debug=True)
