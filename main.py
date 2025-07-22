
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import json
from datetime import datetime
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
def init_db():
    conn = sqlite3.connect('careerlink.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            skills TEXT,
            experience_level TEXT,
            preferred_location TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Jobs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            description TEXT NOT NULL,
            requirements TEXT NOT NULL,
            location TEXT,
            salary_range TEXT,
            job_type TEXT,
            posted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Courses table
    c.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            provider TEXT NOT NULL,
            skill_category TEXT NOT NULL,
            difficulty_level TEXT,
            duration TEXT,
            url TEXT
        )
    ''')
    
    # User progress table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            course_id INTEGER,
            progress INTEGER DEFAULT 0,
            completed BOOLEAN DEFAULT FALSE,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (course_id) REFERENCES courses (id)
        )
    ''')
    
    # Mentorship table
    c.execute('''
        CREATE TABLE IF NOT EXISTS mentorship (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mentor_name TEXT NOT NULL,
            expertise TEXT NOT NULL,
            experience_years INTEGER,
            availability TEXT,
            rating REAL DEFAULT 0.0,
            contact_info TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Sample data insertion
def insert_sample_data():
    conn = sqlite3.connect('careerlink.db')
    c = conn.cursor()
    
    # Sample jobs
    sample_jobs = [
        ("Software Engineer", "Tech Corp", "Develop web applications using Python and JavaScript", "Python, JavaScript, React, SQL", "Remote", "$70,000 - $90,000", "Full-time"),
        ("Data Scientist", "AI Solutions", "Analyze data and build machine learning models", "Python, Machine Learning, Statistics, SQL", "New York", "$80,000 - $120,000", "Full-time"),
        ("Frontend Developer", "WebTech Inc", "Create responsive user interfaces", "HTML, CSS, JavaScript, React, Vue.js", "San Francisco", "$60,000 - $85,000", "Full-time"),
        ("DevOps Engineer", "Cloud Systems", "Manage cloud infrastructure and CI/CD pipelines", "AWS, Docker, Kubernetes, Linux", "Austin", "$75,000 - $100,000", "Full-time"),
        ("UX Designer", "Design Studio", "Design user-centered digital experiences", "Figma, Adobe Creative Suite, User Research", "Los Angeles", "$55,000 - $75,000", "Full-time")
    ]
    
    c.executemany('INSERT OR IGNORE INTO jobs (title, company, description, requirements, location, salary_range, job_type) VALUES (?, ?, ?, ?, ?, ?, ?)', sample_jobs)
    
    # Sample courses with working educational links
    sample_courses = [
        ("AWS Cloud Essentials", "Learn cloud computing fundamentals", "AWS Free Tier", "Cloud", "Beginner", "25 hours", "https://aws.amazon.com/free/"),
        ("UI/UX Design Fundamentals", "Design thinking and user experience", "Google Design", "Design", "Beginner", "35 hours", "https://design.google/")
    ]
    
    c.executemany('INSERT OR IGNORE INTO courses (title, description, provider, skill_category, difficulty_level, duration, url) VALUES (?, ?, ?, ?, ?, ?, ?)', sample_courses)
    
    # Sample mentors
    sample_mentors = [
        ("Sarah Johnson", "Software Engineering, Python, Web Development", 8, "Weekends", 4.8, "sarah.j@email.com"),
        ("Michael Chen", "Data Science, Machine Learning, Statistics", 6, "Evenings", 4.9, "m.chen@email.com"),
        ("Emily Rodriguez", "UX Design, User Research, Product Design", 5, "Flexible", 4.7, "emily.r@email.com"),
        ("David Kim", "DevOps, Cloud Computing, AWS", 7, "Weekdays", 4.6, "david.k@email.com"),
        ("Lisa Thompson", "Frontend Development, React, JavaScript", 4, "Mornings", 4.8, "lisa.t@email.com")
    ]
    
    c.executemany('INSERT OR IGNORE INTO mentorship (mentor_name, expertise, experience_years, availability, rating, contact_info) VALUES (?, ?, ?, ?, ?, ?)', sample_mentors)
    
    conn.commit()
    conn.close()

# AI Job Matching System
class JobMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def match_jobs(self, user_skills, user_experience_level="", preferred_location=""):
        conn = sqlite3.connect('careerlink.db')
        c = conn.cursor()
        c.execute('SELECT * FROM jobs')
        jobs = c.fetchall()
        conn.close()
        
        if not jobs:
            return []
        
        # Combine user profile
        user_profile = f"{user_skills} {user_experience_level} {preferred_location}"
        
        # Combine job requirements and descriptions
        job_texts = []
        for job in jobs:
            job_text = f"{job[3]} {job[4]} {job[5]}"  # description + requirements + location
            job_texts.append(job_text)
        
        # Add user profile to the mix for vectorization
        all_texts = job_texts + [user_profile]
        
        try:
            # Fit vectorizer and transform texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarity between user profile and jobs
            user_vector = tfidf_matrix[-1]  # Last item is user profile
            job_vectors = tfidf_matrix[:-1]  # All except last are jobs
            
            similarities = cosine_similarity(user_vector, job_vectors).flatten()
            
            # Get top matches
            job_matches = []
            for i, similarity in enumerate(similarities):
                job_data = jobs[i]
                job_matches.append({
                    'id': job_data[0],
                    'title': job_data[1],
                    'company': job_data[2],
                    'description': job_data[3],
                    'requirements': job_data[4],
                    'location': job_data[5],
                    'salary_range': job_data[6],
                    'job_type': job_data[7],
                    'match_score': float(similarity)
                })
            
            # Sort by similarity score
            job_matches.sort(key=lambda x: x['match_score'], reverse=True)
            return job_matches[:10]  # Return top 10 matches
            
        except Exception as e:
            print(f"Error in job matching: {e}")
            return []

# Course Recommendation System
class CourseRecommender:
    def recommend_courses(self, user_skills, target_skills=""):
        conn = sqlite3.connect('careerlink.db')
        c = conn.cursor()
        c.execute('SELECT * FROM courses')
        courses = c.fetchall()
        conn.close()
        
        if not courses:
            return []
        
        # Simple skill gap analysis
        user_skill_list = [skill.strip().lower() for skill in user_skills.split(',')]
        target_skill_list = [skill.strip().lower() for skill in target_skills.split(',')]
        
        skill_gaps = []
        for target_skill in target_skill_list:
            if target_skill not in user_skill_list:
                skill_gaps.append(target_skill)
        
        recommendations = []
        for course in courses:
            course_title = course[1].lower()
            course_category = course[4].lower()
            course_description = course[2].lower()
            
            relevance_score = 0
            for gap_skill in skill_gaps:
                if gap_skill in course_title or gap_skill in course_category or gap_skill in course_description:
                    relevance_score += 1
            
            if relevance_score > 0:
                recommendations.append({
                    'id': course[0],
                    'title': course[1],
                    'description': course[2],
                    'provider': course[3],
                    'skill_category': course[4],
                    'difficulty_level': course[5],
                    'duration': course[6],
                    'url': course[7],
                    'relevance_score': relevance_score
                })
        
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        return recommendations[:5]

# Initialize components
job_matcher = JobMatcher()
course_recommender = CourseRecommender()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        skills = request.form.get('skills', '')
        experience_level = request.form.get('experience_level', '')
        preferred_location = request.form.get('preferred_location', '')
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('careerlink.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, email, password_hash, skills, experience_level, preferred_location) VALUES (?, ?, ?, ?, ?, ?)',
                     (username, email, password_hash, skills, experience_level, preferred_location))
            conn.commit()
            conn.close()
            
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('careerlink.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['user_skills'] = user[4]
            session['user_experience'] = user[5]
            session['user_location'] = user[6]
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get job matches
    user_skills = session.get('user_skills', '')
    user_experience = session.get('user_experience', '')
    user_location = session.get('user_location', '')
    
    job_matches = job_matcher.match_jobs(user_skills, user_experience, user_location)
    
    return render_template('dashboard.html', job_matches=job_matches[:5])

@app.route('/job-matches')
def job_matches():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_skills = session.get('user_skills', '')
    user_experience = session.get('user_experience', '')
    user_location = session.get('user_location', '')
    
    matches = job_matcher.match_jobs(user_skills, user_experience, user_location)
    return render_template('job_matches.html', matches=matches)

@app.route('/courses')
def courses():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    target_skills = request.args.get('target_skills', '')
    user_skills = session.get('user_skills', '')
    
    if target_skills:
        recommendations = course_recommender.recommend_courses(user_skills, target_skills)
    else:
        conn = sqlite3.connect('careerlink.db')
        c = conn.cursor()
        c.execute('SELECT * FROM courses')
        courses_data = c.fetchall()
        conn.close()
        
        recommendations = []
        for course in courses_data:
            recommendations.append({
                'id': course[0],
                'title': course[1],
                'description': course[2],
                'provider': course[3],
                'skill_category': course[4],
                'difficulty_level': course[5],
                'duration': course[6],
                'url': course[7]
            })
    
    return render_template('courses.html', courses=recommendations)

@app.route('/cv-assistant')
def cv_assistant():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('cv_assistant.html')

@app.route('/mentorship')
def mentorship():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('careerlink.db')
    c = conn.cursor()
    c.execute('SELECT * FROM mentorship ORDER BY rating DESC')
    mentors = c.fetchall()
    conn.close()
    
    mentor_list = []
    for mentor in mentors:
        mentor_list.append({
            'id': mentor[0],
            'name': mentor[1],
            'expertise': mentor[2],
            'experience_years': mentor[3],
            'availability': mentor[4],
            'rating': mentor[5],
            'contact_info': mentor[6]
        })
    
    return render_template('mentorship.html', mentors=mentor_list)

@app.route('/api/schedule-session', methods=['POST'])
def schedule_session():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    mentor_id = data.get('mentor_id')
    session_date = data.get('session_date')
    session_time = data.get('session_time')
    topic = data.get('topic', '')
    
    # Get mentor details
    conn = sqlite3.connect('careerlink.db')
    c = conn.cursor()
    c.execute('SELECT mentor_name FROM mentorship WHERE id = ?', (mentor_id,))
    mentor = c.fetchone()
    conn.close()
    
    if not mentor:
        return jsonify({'error': 'Mentor not found'}), 404
    
    # In a real application, you would store this in a sessions table
    # For now, we'll just return a success message
    return jsonify({
        'success': True,
        'message': f'Session scheduled with {mentor[0]} on {session_date} at {session_time}',
        'mentor_name': mentor[0],
        'date': session_date,
        'time': session_time,
        'topic': topic
    })

@app.route('/api/send-message', methods=['POST'])
def send_message():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    mentor_id = data.get('mentor_id')
    message = data.get('message', '')
    
    # Get mentor details
    conn = sqlite3.connect('careerlink.db')
    c = conn.cursor()
    c.execute('SELECT mentor_name, contact_info FROM mentorship WHERE id = ?', (mentor_id,))
    mentor = c.fetchone()
    conn.close()
    
    if not mentor:
        return jsonify({'error': 'Mentor not found'}), 404
    
    # In a real application, you would store this in a messages table
    # For now, we'll just return a success message
    return jsonify({
        'success': True,
        'message': f'Message sent to {mentor[0]}',
        'mentor_name': mentor[0],
        'mentor_email': mentor[1],
        'user_message': message
    })

@app.route('/api/get-mentor-details/<int:mentor_id>')
def get_mentor_details(mentor_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    conn = sqlite3.connect('careerlink.db')
    c = conn.cursor()
    c.execute('SELECT * FROM mentorship WHERE id = ?', (mentor_id,))
    mentor = c.fetchone()
    conn.close()
    
    if not mentor:
        return jsonify({'error': 'Mentor not found'}), 404
    
    mentor_data = {
        'id': mentor[0],
        'name': mentor[1],
        'expertise': mentor[2],
        'experience_years': mentor[3],
        'availability': mentor[4],
        'rating': mentor[5],
        'contact_info': mentor[6]
    }
    
    return jsonify(mentor_data)

@app.route('/api/cv-feedback', methods=['POST'])
def cv_feedback():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    cv_text = request.json.get('cv_text', '')
    feedback = analyze_cv_text(cv_text)
    return jsonify({'feedback': feedback})

@app.route('/api/cv-upload-feedback', methods=['POST'])
def cv_upload_feedback():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'cv_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['cv_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from file
            cv_text = extract_text_from_file(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Analyze the extracted text
            feedback = analyze_cv_text(cv_text)
            return jsonify({'feedback': feedback})
            
        except Exception as e:
            # Clean up file if error occurs
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def extract_text_from_file(filepath):
    """Extract text from uploaded file"""
    filename = filepath.lower()
    
    if filename.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    elif filename.endswith('.pdf'):
        # For PDF files, we'll provide a simplified extraction
        # In a production environment, you'd want to use libraries like PyPDF2 or pdfplumber
        try:
            # Simple PDF text extraction placeholder
            # You would need to install PyPDF2: pip install PyPDF2
            # import PyPDF2
            # with open(filepath, 'rb') as f:
            #     reader = PyPDF2.PdfReader(f)
            #     text = ""
            #     for page in reader.pages:
            #         text += page.extract_text()
            #     return text
            
            # For now, return a message indicating PDF processing
            return "PDF file uploaded. For full PDF processing, additional libraries would be needed in production."
        except:
            return "Error reading PDF file. Please try converting to text format."
    
    elif filename.endswith(('.doc', '.docx')):
        # For Word documents, you'd typically use python-docx
        # For now, return a placeholder message
        return "Word document uploaded. For full Word document processing, additional libraries would be needed in production."
    
    else:
        return "Unsupported file format."

def analyze_cv_text(cv_text):
    """Analyze CV text and provide feedback"""
    feedback = []
    
    if len(cv_text) < 200:
        feedback.append("Your CV seems quite short. Consider adding more detail about your experience and skills.")
    
    if 'python' in cv_text.lower() or 'javascript' in cv_text.lower():
        feedback.append("Great! You have programming skills mentioned. Consider adding specific projects you've worked on.")
    
    if '@' not in cv_text:
        feedback.append("Make sure to include your email address for contact purposes.")
    
    if not any(keyword in cv_text.lower() for keyword in ['project', 'developed', 'created', 'built']):
        feedback.append("Consider adding more action verbs and specific projects to showcase your achievements.")
    
    if 'education' not in cv_text.lower() and 'degree' not in cv_text.lower():
        feedback.append("Consider adding your educational background if relevant to your target position.")
    
    if 'experience' not in cv_text.lower() and 'work' not in cv_text.lower():
        feedback.append("Make sure to highlight your work experience and professional background.")
    
    if not feedback:
        feedback.append("Your CV looks good! Keep updating it with new skills and experiences.")
    
    return feedback

@app.route('/api/interview-questions')
def interview_questions():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    job_role = request.args.get('role', 'Software Developer')
    
    # Sample interview questions based on role
    questions_db = {
        'Software Developer': [
            "Tell me about yourself and your programming background.",
            "Explain the difference between procedural and object-oriented programming.",
            "How do you handle debugging in your code?",
            "Describe a challenging project you've worked on.",
            "What programming languages are you most comfortable with?"
        ],
        'Data Scientist': [
            "How do you approach a new data science problem?",
            "Explain the bias-variance tradeoff in machine learning.",
            "What's your experience with data visualization tools?",
            "How do you handle missing data in datasets?",
            "Describe a machine learning project you've completed."
        ]
    }
    
    questions = questions_db.get(job_role, questions_db['Software Developer'])
    return jsonify({'questions': questions})

if __name__ == '__main__':
    init_db()
    insert_sample_data()
    app.run(host='0.0.0.0', port=5000, debug=True)
