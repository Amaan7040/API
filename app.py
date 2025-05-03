from flask import Flask, jsonify, render_template, request, session, send_file, redirect, url_for
from io import BytesIO
from reportlab.pdfgen import canvas
import json
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
from werkzeug.security import generate_password_hash, check_password_hash 
from dotenv import load_dotenv
import os

load_dotenv("credentials.env")

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
    elif "diploma" in text_lower:
        return "Diploma"
    else:
        return "Other"

def count_certifications(resume_text):
    return len(re.findall(r"certified|certification|certificate", resume_text, re.IGNORECASE))

def count_projects(resume_text):
    return len(re.findall(r"project", resume_text, re.IGNORECASE))

# Updated education mapping:
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
    # If your combined model is only for job role prediction,
    # you may keep this separate resume scoring function as-is.
    # Otherwise, if the combined model also provides scoring, 
    # update this function accordingly.
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
    "devops", "image processing", "jira", "postman", "excel", "data processing", "scikit-learn",
    "matplotlib", "seaborn", "api integration", "data mining","scikit-learn", "data preprocessing"
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
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
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

# ==========================================================
# Updated: Load combined model and vectorizer (tvectorizer)
# ==========================================================
def load_model_and_vectorizer():
    try:
        with open("combined_job_predict_model1.pkl", "rb") as model_file:
            combined_model = pickle.load(model_file)
        with open("combined_tfidf_vectorizer1.pkl", "rb") as vectorizer_file:
            tvectorizer = pickle.load(vectorizer_file)
        print("[DEBUG] Combined model and vectorizer loaded.")
        return combined_model, tvectorizer
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None, None

# ---------------------- Main Resume Processing ---------------------- #

def process_resume(file):
    # Load the new combined model and vectorizer
    rf, vectorizer = load_model_and_vectorizer()
    if not rf or not vectorizer:
        return "ML model missing!", None, [], None, "india", None, None
    
    text = extract_text_from_file(file)
    if not text:
        return "No readable text!", None, [], None, "india", None, None
    
    user_name = extract_name(text)
    extracted_skills = extract_skills(text)
    resume_country = "india"
    
    try:
        text_vectorized = vectorizer.transform([text])
        print(f"[DEBUG] Non-zero elements: {text_vectorized.nonzero()}")
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
app.secret_key=os.getenv("FLASK_SECRET_KEY")

@app.route("/upload-chart", methods=["POST"])
def upload_chart():
    data = request.get_json()
    chart_base64 = data.get("chart", "")

    if chart_base64.startswith("data:image/png;base64,"):
        chart_base64 = chart_base64.split(",")[1]

    session["chart_image"] = chart_base64
    return jsonify({"status": "received"})

@app.route("/download-report")
def download_report():
    data = session.get('report_data')
    if not data:
        return redirect(url_for('index'))

    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 800, "Resume Analysis Report")
    p.setFont("Helvetica", 12)

    y = 760
    for label, value in [
        ("Name:", data["name"]),
        ("Predicted Role:", data["predicted_job"]),
        ("Score:", f"{data['resume_score']} ({data['resume_category']})"),
    ]:
        p.drawString(50, y, f"{label} {value}")
        y -= 20

    # Skills
    p.drawString(50, y, "Extracted Skills:")
    y -= 20
    for skill in data["skills"]:
        p.drawString(70, y, f"- {skill}")
        y -= 15
        if y < 50:
            p.showPage()
            y = 800

    # Missing Skills
    if data["missing_skills"]:
        p.drawString(50, y, "Recommended (Missing) Skills:")
        y -= 20
        for skill in data["missing_skills"]:
            p.drawString(70, y, f"- {skill}")
            y -= 15
            if y < 50:
                p.showPage()
                y = 800

    # Add radar chart image if available
    chart_data = session.get("chart_image")
    if chart_data:
        try:
            from base64 import b64decode
            from reportlab.lib.utils import ImageReader
            from PIL import Image
            img_data = b64decode(chart_data)
            img = Image.open(BytesIO(img_data))
            img_io = BytesIO()
            img.save(img_io, format='PNG')
            img_io.seek(0)
            p.drawImage(ImageReader(img_io), 100, 400, width=400, height=300)
        except Exception as e:
            print("[ERROR] Failed to draw chart:", e)

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="resume_analysis_report.pdf",
        mimetype="application/pdf"
    )

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
                    # ✅ Store for PDF report
                    session['report_data'] = {
                        "name": user_name or "Unknown",
                        "predicted_job": predicted_job or "N/A",
                        "resume_score": resume_score,
                        "resume_category": resume_category,
                        "skills": extracted_skills,
                        "missing_skills": missing_skills
                    }
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
        "x-rapidapi-key": os.getenv("API_KEY"),
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }
    params = {"query": f"{query} in {country}", "page": page, "num_pages": 1}
    if job_type:
        params["job_type"] = job_type
    response = requests.get(url, headers=headers, params=params)
    return response.text

@app.route("/api/skill-match-data", methods=["POST"])
def skill_match_data():
    data = request.get_json()
    predicted_job = data.get("job_role")
    extracted_skills = data.get("extracted_skills", [])

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT skills FROM jobrolesskills WHERE job_role = %s", (predicted_job,))
        job_data = cursor.fetchone()
        cursor.close()
        conn.close()

        if job_data:
            required_skills = job_data["skills"].lower().split(", ")
            required_set = set(required_skills)
            extracted_set = set(s.lower() for s in extracted_skills)

            labels = required_skills[:10]  
            match_values = [1 if skill in extracted_set else 0 for skill in labels]

            return jsonify({
                "labels": labels,
                "values": match_values,
            })
        else:
            return jsonify({"labels": [], "values": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    message = ""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if not (username and email and password):
            message = "All fields are required!"
            return render_template('signup.html', message=message)
        hashed_password = generate_password_hash(password)
        try:
            conn = get_db_connection("signup_db")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users_signup_details WHERE email = %s", (email,))
            if cursor.fetchone():
                message = "User already exists! Try logging in."
            else:
                cursor.execute(
                    "INSERT INTO users_signup_details (username, email, password) VALUES (%s, %s, %s)",
                    (username, email, hashed_password)
                )
                conn.commit()
                message = "Signup Successful! Now you can login."
            cursor.close()
            conn.close()
        except Exception as e:
            message = f"Error: {e}"
    return render_template('signup.html', message=message)

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            conn = get_db_connection("signup_db")
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, password FROM users_signup_details WHERE email = %s", (email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            if user:
                user_id, username, stored_password = user
                if check_password_hash(stored_password, password):
                    session['user_id'] = user_id
                    session['username'] = username
                    return redirect('/dashboard')
                else:
                    message = "Invalid Credentials!"
            else:
                message = "Invalid Credentials!"
        except Exception as e:
            message = f"Error: {e}"
    return render_template('login.html', message=message)

@app.route('/dashboard')
def dashboard():
    # if 'username' in session:
    #     return f"Welcome {session['username']} to your dashboard!"
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
