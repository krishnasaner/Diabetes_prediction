from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained pipeline (includes scaler + model)
model = pickle.load(open("Diabetes.pkl", "rb"))

# Feature metadata for validation
FEATURES = [
    {
        'name': 'Pregnancies', 'key': '1', 'id': 'pregnancies',
        'placeholder': 'e.g. 2', 'min': 0, 'max': 20, 'step': 1,
        'icon': '👶', 'description': 'Number of times pregnant'
    },
    {
        'name': 'Glucose', 'key': '2', 'id': 'glucose',
        'placeholder': 'e.g. 120', 'min': 44, 'max': 200, 'step': 1,
        'icon': '🩸', 'description': 'Plasma glucose concentration (mg/dL)'
    },
    {
        'name': 'Blood Pressure', 'key': '3', 'id': 'bloodpressure',
        'placeholder': 'e.g. 72', 'min': 24, 'max': 122, 'step': 1,
        'icon': '💓', 'description': 'Diastolic blood pressure (mm Hg)'
    },
    {
        'name': 'Skin Thickness', 'key': '4', 'id': 'skinthickness',
        'placeholder': 'e.g. 20', 'min': 0, 'max': 99, 'step': 1,
        'icon': '📏', 'description': 'Triceps skin fold thickness (mm)'
    },
    {
        'name': 'Insulin', 'key': '5', 'id': 'insulin',
        'placeholder': 'e.g. 80', 'min': 0, 'max': 846, 'step': 1,
        'icon': '💉', 'description': '2-Hour serum insulin (mu U/ml)'
    },
    {
        'name': 'BMI', 'key': '6', 'id': 'bmi',
        'placeholder': 'e.g. 32.0', 'min': 10, 'max': 70, 'step': 0.1,
        'icon': '⚖️', 'description': 'Body mass index (kg/m²)'
    },
    {
        'name': 'Diabetes Pedigree', 'key': '7', 'id': 'dpf',
        'placeholder': 'e.g. 0.5', 'min': 0.05, 'max': 2.5, 'step': 0.001,
        'icon': '🧬', 'description': 'Diabetes pedigree function (hereditary risk)'
    },
    {
        'name': 'Age', 'key': '8', 'id': 'age',
        'placeholder': 'e.g. 30', 'min': 18, 'max': 100, 'step': 1,
        'icon': '🎂', 'description': 'Age in years'
    },
]

COLUMN_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

def get_risk_level(probability):
    """Categorize risk based on probability."""
    if probability >= 0.75:
        return {'level': 'High Risk', 'color': '#ef4444', 'advice': 'Consult a healthcare professional immediately.'}
    elif probability >= 0.50:
        return {'level': 'Moderate Risk', 'color': '#f59e0b', 'advice': 'Schedule a check-up with your doctor soon.'}
    elif probability >= 0.30:
        return {'level': 'Low Risk', 'color': '#3b82f6', 'advice': 'Maintain a healthy lifestyle and monitor regularly.'}
    else:
        return {'level': 'Minimal Risk', 'color': '#10b981', 'advice': 'Keep up the good work with your health habits!'}

def get_feature_insights(values):
    """Provide insight for each feature value."""
    insights = []
    thresholds = {
        'Glucose': (70, 140, 'Normal fasting glucose: 70-100 mg/dL'),
        'BloodPressure': (60, 90, 'Normal diastolic BP: 60-80 mm Hg'),
        'BMI': (18.5, 30, 'Normal BMI: 18.5-24.9 kg/m²'),
        'Age': (0, 45, 'Risk increases significantly after age 45'),
    }
    for i, col in enumerate(COLUMN_NAMES):
        if col in thresholds:
            low, high, note = thresholds[col]
            val = values[i]
            if val > high:
                insights.append({'feature': col, 'status': 'elevated', 'note': note})
            elif val < low:
                insights.append({'feature': col, 'status': 'low', 'note': note})
            else:
                insights.append({'feature': col, 'status': 'normal', 'note': note})
    return insights


@app.route('/')
def home():
    return render_template("index.html", features=FEATURES)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = []
        for feat in FEATURES:
            val = request.form.get(feat['key'], '').strip()
            if not val:
                return render_template('index.html', features=FEATURES,
                                       error=f"Please fill in the {feat['name']} field.")
            values.append(float(val))
    except ValueError:
        return render_template('index.html', features=FEATURES,
                               error='Please enter valid numeric values for all fields.')

    row_df = pd.DataFrame([values], columns=COLUMN_NAMES)

    try:
        prediction = model.predict_proba(row_df)
        prob = float(prediction[0][1])
        risk = get_risk_level(prob)
        insights = get_feature_insights(values)

        return render_template('index.html',
                               features=FEATURES,
                               result=True,
                               probability=round(prob * 100, 1),
                               risk=risk,
                               insights=insights,
                               values={feat['key']: values[i] for i, feat in enumerate(FEATURES)})
    except Exception as e:
        return render_template('index.html', features=FEATURES,
                               error=f'Prediction error: {e}')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for programmatic access."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        values = [float(data.get(col, 0)) for col in COLUMN_NAMES]
        row_df = pd.DataFrame([values], columns=COLUMN_NAMES)

        prediction = model.predict_proba(row_df)
        prob = float(prediction[0][1])
        risk = get_risk_level(prob)

        return jsonify({
            'probability': round(prob * 100, 1),
            'risk_level': risk['level'],
            'advice': risk['advice'],
            'prediction': 'Diabetic' if prob > 0.5 else 'Non-Diabetic'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting the Flask application...")
    app.run(host='127.0.0.1', port=8080, debug=True)