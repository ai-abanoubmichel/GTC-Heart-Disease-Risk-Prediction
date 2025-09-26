import pandas as pd 
import numpy as np 
import streamlit as st 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
import joblib

# ================================
# Load dataset
# ================================
Heart_disease = pd.read_csv('https://raw.githubusercontent.com/ai-abanoubmichel/GTC-Heart-Disease-Risk-Prediction/refs/heads/main/Datasets/Heart_disease_cleveland_new.csv') 

# Set binary valued columns as bool
Heart_disease['sex'] = Heart_disease['sex'].astype(bool)
Heart_disease['fbs'] = Heart_disease['fbs'].astype(bool)
Heart_disease['exang'] = Heart_disease['exang'].astype(bool)
Heart_disease['target'] = Heart_disease['target'].astype(bool)

# ================================
# Feature engineering functions
# ================================
def categorize_age(age):
    if age < 40:
        return 'Young'
    elif age < 50:
        return 'Middle-aged'
    elif age < 60:
        return 'Mature'
    else:
        return 'Senior'

def categorize_bp(bp):
    if bp < 120:
        return 'Normal'
    elif bp < 140:
        return 'High'
    else:
        return 'Very High'

def categorize_chol(chol):
    if chol < 200:
        return 'Good'
    elif chol < 240:
        return 'Borderline'
    else:
        return 'High'

def calculate_risk_score(row):
    score = 0
    if row['age'] > 60:
        score += 1
    if row['trestbps'] > 140:
        score += 1
    if row['chol'] > 240:
        score += 1
    if row['exang']:
        score += 1
    return score

def feature_engineering_func(df):
    df = df.copy()
    df['age_group'] = df['age'].apply(categorize_age)
    df['bp_category'] = df['trestbps'].apply(categorize_bp)
    df['chol_category'] = df['chol'].apply(categorize_chol)
    df['risk_score'] = df.apply(calculate_risk_score, axis=1)
    return df

# ================================
# Load or Train Model 
# ================================
@st.cache_resource
def load_model():
    try:
        clf = joblib.load("heart_disease_pipeline.pkl")
        return clf
    except:
        X = Heart_disease.drop("target", axis=1)
        y = Heart_disease["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Preprocessing
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 
                           'ca', 'thal', 'age_group', 'bp_category', 'chol_category']
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'risk_score']
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ])
        
        # Pipeline
        clf = Pipeline(steps=[
            ('feature_engineering', FunctionTransformer(feature_engineering_func)),
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='poly', gamma='scale', degree=3, C=1, probability=True))
        ])
        
        # Train model
        clf.fit(X_train, y_train)
        
        # Save pipeline
        joblib.dump(clf, "heart_disease_pipeline.pkl")
        
        return clf


# Load pipeline
clf = load_model()

#set the browser tab title
st.set_page_config(page_title="Heart Disease Prediction" ,layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.southcoasthealth.com/assets/upload/695693f2-a879-4a36-aab5-386916191104/heart-attack-concept.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""

# Inject the CSS
st.markdown(page_bg, unsafe_allow_html=True)

#streamlit Web title
st.title("❤️ Heart Disease Prediction App")

#tell the user to enter the patient data 
st.write("Enter your data to know your risk of developing heart disease:")

# ================================
# User Inputs
# ================================
col1, col2 = st.columns(2)
with col1: 
    age = st.number_input("Age", min_value=20, max_value=100, value=45)
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    slope = st.selectbox("Slope of the peak exercise ST segment(slope)", [0, 1, 2])

with col2:
    sex = st.selectbox("Sex(0 = Female , 1 = Male)", [0, 1])  # 0 = Female, 1 = Male
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=60, max_value=220, value=150)
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ca = st.selectbox("Number of major vessels (0–3) colored by flourosopy(ca)", [0, 1, 2, 3])
    
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# ================================
# Prepare input for prediction
# ================================
input_data = pd.DataFrame([{
    'age': age,
    'cp': cp,
    'chol': chol,
    'restecg': restecg,
    'exang': bool(exang),
    'slope': slope,
    'sex': bool(sex),
    'trestbps': trestbps,
    'fbs': bool(fbs),
    'thalach': thalach,
    'oldpeak': oldpeak,
    'ca': ca,
    'thal': thal
}])

# ================================
# Prediction
# ================================
if st.button("🔍 Predict"):
    prediction = clf.predict(input_data)[0]
    if prediction:
        st.error("⚠️ High Risk: The model predicts a strong likelihood of heart disease.")
    else:
        st.success("✅ Low Risk: No strong indication of heart disease detected.")
