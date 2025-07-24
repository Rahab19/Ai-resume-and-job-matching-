# Ai-Resume and Job Matching Web Application

## Overview

This project is a Python-based web application designed to help users with resume processing and job matching. It leverages AI/machine learning capabilities (specifically Scikit-learn for text processing) to provide intelligent functionalities. The application is built with Flask and uses Gunicorn for serving, with a SQLite database (`careerlink.db`) for data storage.

---

## Features

-  **User Authentication** – Secure login & signup with sessions/JWT.  
-  **AI Job Matching** – Upload or paste your resume and get ranked job recommendations.  
-  **Course Recommendation System** – AI suggests online courses for skill gaps.  
-  **AI CV & Interview Assistant** – Get instant feedback on your CV and mock interview Q&A.  
-  **Human Mentorship Dashboard** – Connect with mentors and track your growth.  
-  **Job Board Integration** – Scrapes or fetches job listings dynamically.

---

## Technologies Used

* **Python**
* **Flask** (Web Framework)
* **Gunicorn** (WSGI HTTP Server)
* **Scikit-learn** (Machine Learning Library for `TfidfVectorizer`)
* **SQLite** (`careerlink.db` for database)
* **HTML/CSS/JavaScript** (for the front-end)


---

##  Getting Started

###  Installation

1. **Clone the repo**
```bash
git clone https://github.com/Rahab19/Ai-resume-and-job-matching-.git
cd Ai-resume-and-job-matching-
```

2. **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application Locally

To start the Flask development server:

```bash
python main.py
```

### Live link
```bash
https://ai-resume-and-job-matching.onrender.com
```

## Deployment
This application is designed to be deployed on a platform that supports Python web applications.


