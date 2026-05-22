---

# Insulix — AI-Powered Diabetes Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.3-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?style=flat-square&logo=scikit-learn)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat-square&logo=render)
![Status](https://img.shields.io/badge/Status-Live-success?style=flat-square)

> An AI-powered clinical tool that predicts diabetes risk from 
> health indicators using machine learning.

🔗 **Live Demo:** https://diabetes-prediction-p2an.onrender.com

---

## 🧠 What is Insulix?

Insulix is an end-to-end machine learning web application that 
predicts whether a patient is at risk of diabetes based on 
clinical health indicators. Built using Flask and scikit-learn, 
it takes 8 medical inputs and returns an instant risk assessment 
powered by a trained ML model.

---

## ✨ Features

- AI-powered risk prediction using scikit-learn
- Clean, responsive UI with dark clinical theme
- 8 health parameter inputs (Glucose, BMI, Age, etc.)
- Instant result with Low / Medium / High risk classification
- Deployed live on Render

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| ML Model | scikit-learn (Random Forest / Logistic Regression) |
| Dataset | Pima Indians Diabetes Dataset |
| Deployment | Render |

---

## 📁 Project Structure 
```text
Diabetes_prediction/
├── Diabetes_Prediction-Project-Code/
│   ├── app.py              # Flask application
│   ├── Diabetes.pkl        # Trained ML model
│   ├── requirements.txt    # Python dependencies
│   ├── templates/          # HTML templates
│   └── static/             # CSS, JS, assets
├── .gitignore
├── README.md
└── render.yaml             # Render deployment config
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/krishnasaner/Diabetes_prediction.git
cd Diabetes_prediction/Diabetes_Prediction-Project-Code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Visit `http://localhost:5000`

---

## 📊 Model Info

- **Dataset:** Pima Indians Diabetes Dataset (768 samples, 8 features)
- **Target:** Binary classification (Diabetic / Non-Diabetic)
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness,
  Insulin, BMI, Diabetes Pedigree Function, Age

---

## ⚠️ Disclaimer

This tool is built for **educational purposes only** and is not 
intended for medical diagnosis. Always consult a healthcare 
professional for medical advice.

---

## 👨💻 Author

**Krishna Saner**  
B.Tech Computer Science — DY Patil University, Pune  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/krishnasaner)

---

*Built with scikit-learn & Flask • Pima Indians Diabetes Dataset*

---
