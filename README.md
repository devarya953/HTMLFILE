# NeuroScan - AI-Powered Brain Tumor Detection System

NeuroScan is a smart medical tool that uses AI (VGG19 model) to detect brain tumors in MRI scans. It provides fast, accurate results through a secure and intuitive web interface.

## Features
- AI tumor detection (VGG19)
- Patient record management
- Secure professional login
- MRI scan upload & analysis
- Auto PDF report generation
- MySQL-backed storage

## Tech Stack
- Frontend: HTML, CSS, JS, Animate.css, Font Awesome, html2pdf.js
- Backend: Python (Flask, Flask-CORS), TensorFlow/Keras, MySQL
- Image Processing: OpenCV, NumPy

## Folder Structure
- `frontend/` → login.html, Dashboard.html, report.html, style.css
- `backend/` → app.py, vgg19_model_03.h5
- `database/` → MySQL scripts
- `README.md`

## Setup Instructions

### Prerequisites
- Python 3.8+
- MySQL Server
- Node.js

### 1. Clone the Repo
```bash
https://github.com/Raman1717/Brain-Tumor-Detection
```

### 2. Backend Setup
```bash
cd NeuroScan/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install flask flask-cors tensorflow opencv-python mysql-connector-python
```

- Create a MySQL database named `BrainTumorDB`
- Update DB credentials in `app.py`
- Run the app:
```bash
python app.py
```

### 3. Frontend Setup
```bash
cd ../frontend
python -m http.server 8000
```
Open in browser: `http://localhost:8000/login.html`

## Usage
- Login: `abc / abc@123`
- Enter patient info (name, phone, age, blood type)
- Upload MRI image → Click "Analyze"
- View results → Generate PDF report

## Security Features
- Disabled right-click & dev tools
- Login rate limit (5/min)
- Session timeout (5 mins)
- CSRF protection, CSP, strong password rules

## API Endpoint

### POST /predict
**Form Data:**
- `file`: MRI image file
- `name`: Patient name
- `phn`: Phone number
- `age`: Age
- `bloodType`: Blood type

**Response:** Tumor result + confidence level

---

© 2025 NeuroScan | Precision Diagnosis Powered by AI
