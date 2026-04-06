import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
data = pd.read_csv("diabetes.csv")

print(f"Dataset shape: {data.shape}")
print(f"Class distribution:\n{data['Outcome'].value_counts()}\n")

# ── Data Cleaning ──────────────────────────────────────────────────────
# Columns where 0 is physiologically impossible → treat as missing
zero_impute_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_impute_cols:
    data[col] = data[col].replace(0, np.nan)

# Impute missing values with the median (robust to outliers)
for col in zero_impute_cols:
    median_val = data[col].median()
    data[col].fillna(median_val, inplace=True)
    print(f"  Imputed {col} zeros with median = {median_val}")

print()

# Prepare features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model Comparison ──────────────────────────────────────────────────
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model_name = None
best_score = 0
best_pipeline = None

print("── Cross-Validation Results ─────────────────────────────────")
for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    print(f"  {name:25s}  Accuracy: {mean_score:.4f} (+/- {scores.std():.4f})")
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_pipeline = pipeline

print(f"\n  Best model: {best_model_name} (CV accuracy: {best_score:.4f})\n")

# ── Train best model on full training set ─────────────────────────────
best_pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("── Test Set Evaluation ──────────────────────────────────────")
print(f"  Test accuracy: {test_accuracy:.4f}\n")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save the pipeline (scaler + model) ────────────────────────────────
with open('Diabetes.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

print(f"\nModel pipeline saved to Diabetes.pkl")
print(f"Features expected: {list(X.columns)}")
