# Insulix — AI-Powered Diabetes Risk Assessment

[![Live Demo on Render](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render)](https://diabetes-prediction-p2an.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)]()

**Insulix** is a modern, clinical-grade web application that leverages machine learning to predict the risk of diabetes based on standard health indicators. It is built with a robust Scikit-Learn pipeline and a lightweight Flask backend, featuring a beautiful, professional UI.

🌍 **Live Application:** [https://diabetes-prediction-p2an.onrender.com](https://diabetes-prediction-p2an.onrender.com)

---

## ✨ Features

- **Clinical-Grade UI:** A clean, professional, and accessible design focused on a premium user experience.
- **Robust ML Pipeline:** Utilizes Scikit-Learn pipelines (`StandardScaler` + `RandomForestClassifier` / `GradientBoostingClassifier`).
- **Smart Imputation:** Handles zero-values in biological metrics (e.g., Blood Pressure, Glucose, BMI) correctly using median imputation.
- **Real-Time Assessment:** Instantly computes probability, categorizes risk (Minimal, Low, Moderate, High), and provides actionable clinical insights.
- **REST API:** Features an `/api/predict` endpoint for programmatic access.
- **Production-Ready:** Pre-configured for seamless deployment to [Render](https://render.com) and [Vercel](https://vercel.com).

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, Vanilla JavaScript (Custom SVG Icons, Responsive Design)
- **Backend:** Python, Flask, Gunicorn
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Deployment:** Render (via `render.yaml` & `Procfile`), Vercel (via `vercel.json`)

---

## 🚀 Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/krishnasaner/Diabetes_prediction.git
   cd Diabetes_prediction
   ```

2. **Set up a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r Diabetes_Prediction-Project-Code/requirements.txt
   ```

4. **Run the application:**
   ```bash
   cd Diabetes_Prediction-Project-Code
   python app.py
   ```
   The app will be available at `http://localhost:8080`.

---

## 🤖 API Usage

You can also use the `/api/predict` endpoint for programmatic predictions.

**Request:**
```bash
curl -X POST https://diabetes-prediction-p2an.onrender.com/api/predict \
-H "Content-Type: application/json" \
-d '{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 72,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}'
```

**Response:**
```json
{
    "advice": "Maintain a healthy lifestyle and continue regular monitoring.",
    "prediction": "Non-Diabetic",
    "probability": 34.5,
    "risk_level": "Low Risk"
}
```

---

## 📁 Repository Structure

```text
Diabetes_prediction/
├── Diabetes_Prediction-Project-Code/
│   ├── app.py                  # Main Flask application and API routes
│   ├── train_model.py          # ML training script with imputation pipeline
│   ├── Diabetes.pkl            # Serialized trained Scikit-Learn model
│   ├── diabetes.csv            # Pima Indians Diabetes Dataset
│   ├── requirements.txt        # Python dependencies
│   └── templates/
│       └── index.html          # Clinical UI Template
├── .python-version             # Python version pinning (3.10)
├── vercel.json                 # Vercel Serverless routing config
├── render.yaml                 # Render Blueprint configuration
├── Procfile                    # Render WSGI entry point (gunicorn)
├── Readme.md                   # Project documentation
└── LICENSE                     # MIT License
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*Disclaimer: This tool is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment.*
