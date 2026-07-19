# Insulix — AI-Powered Diabetes Risk Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.3-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?style=flat-square&logo=scikit-learn)
![GitHub Pages](https://img.shields.io/badge/Deployed%20on-GitHub%20Pages-22C55E?style=flat-square&logo=github)
![Status](https://img.shields.io/badge/Status-Live-success?style=flat-square)

> An AI-powered clinical screening tool that predicts diabetes risk from health indicators using cross-validated machine learning.

🔗 **Flask Web App (Render):** [https://diabetes-prediction-p2an.onrender.com](https://diabetes-prediction-p2an.onrender.com)  
🔗 **Static Web App (GitHub Pages):** [https://krishnasaner.github.io/Diabetes_prediction/](https://krishnasaner.github.io/Diabetes_prediction/)

---

## 🧠 What is Insulix?

Insulix is a high-fidelity, end-to-end machine learning web application that predicts whether a patient is at risk of diabetes based on 8 clinical health parameters. Designed with a Swiss-modernist, high-contrast clinical dark theme, it bridges advanced machine learning pipelines with production-grade UI engineering.

The project is deployed in two parallel architectures:
1. **Dynamic Backend (Flask + Render):** Serves predictions via a serialized scikit-learn model on a Flask web server, complete with a REST API endpoint.
2. **Static Frontend (JavaScript + GitHub Pages):** Performs fully client-side ML inference by embedding the StandardScaler metrics and Logistic Regression coefficients directly in the browser's JavaScript engine (zero-server execution).

---

## ✨ Features

- **Clinical Risk Intelligence:** Real-time diabetes risk assessment with probability scoring (0% – 100%) and stratification into Minimal, Low, Moderate, or High Risk.
- **Client-Side Inference:** Fully portabilized Logistic Regression model running directly in browser JavaScript, eliminating server latency and maintaining complete privacy.
- **Editorial UX/UI:** Production-ready design system built with sharp modernist lines, precise typography (Inter), customized status-colored SVG gauges, and linear transitions.
- **Clinical Insight Table:** Compares patient measurements to clinical normal reference ranges (Glucose, BP, BMI, etc.) and highlights elevated indicators.
- **Interactive API Documentation:** Complete programmatic REST API reference showing structured JSON request/response formats and `curl` commands.
- **Semantic HTML & Accessibility:** Screen-reader friendly structures, skip-to-content links, ARIA landmark roles, and WCAG AA color contrast ratios.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Semantic HTML5, Vanilla CSS3 (Modernist Custom Tokens), Vanilla JavaScript |
| **Backend** | Python 3, Flask, Gunicorn |
| **Machine Learning** | scikit-learn (Logistic Regression, Stratified 5-Fold CV, StandardScaler) |
| **Dataset** | Pima Indians Diabetes Dataset (768 patient samples) |
| **Deployment** | GitHub Pages (Static Client-Side), Render (Dynamic Backend Server) |

---

## 📁 Project Structure

```text
Diabetes_prediction/
├── docs/
│   └── index.html          # Static site deployed to GitHub Pages (Client-side ML)
├── Diabetes_Prediction-Project-Code/
│   ├── app.py              # Flask backend server & routes (includes REST API)
│   ├── train_model.py      # ML training and cross-validation script
│   ├── diabetes.csv        # Pima Indians Diabetes Dataset
│   ├── Diabetes.pkl        # Serialized pipeline (scaler + Logistic Regression)
│   ├── requirements.txt    # Python package dependencies
│   └── templates/
│       └── index.html      # Jinja2 Flask HTML template (matching redesign)
├── render.yaml             # Render infrastructure deployment config
├── vercel.json             # Vercel deployment config
└── README.md               # Documentation
```

---

## 🚀 How to Run Locally

### 1. Run the Flask Web Application
```bash
# Clone the repository
git clone https://github.com/krishnasaner/Diabetes_prediction.git
cd Diabetes_prediction/Diabetes_Prediction-Project-Code

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```
Open `http://localhost:8080` in your browser.

### 2. Run the Static Web Application
To run the static frontend version, simply open the file at `docs/index.html` directly in any web browser, or serve it using Python's built-in HTTP server:
```bash
cd docs/
python -m http.server 8080
```
Open `http://localhost:8080` in your browser.

---

## 📊 Model Information

- **Dataset:** Pima Indians Diabetes Dataset (768 female patient samples of Pima Indian heritage).
- **Target:** Classify patients as Diabetic (1) or Non-Diabetic (0).
- **Pipeline:** Imputes physiologically impossible zero-values (in Glucose, BP, BMI, etc.) with training-set medians, normalizes parameters with `StandardScaler`, and evaluates using a Logistic Regression classifier.
- **Accuracy:** Auto-selected via Stratified 5-Fold Cross-Validation comparing Logistic Regression, Random Forest, and Gradient Boosting.

---

## ⚠️ Disclaimer

This tool is built for **educational and screening purposes only**. It does not constitute medical advice or replace professional diagnostic procedures. Always consult a qualified healthcare professional for clinical decisions.

---

## 👨‍💻 Author

**Krishna Saner**  
B.Tech Computer Science — DY Patil University, Pune  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/krishnasaner)
