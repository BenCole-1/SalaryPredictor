import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    with open("feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols

model, feature_cols = load_model()

st.title("💼 Data Scientist Salary Predictor")
st.markdown("Enter your details below to receive a salary prediction:")

# ----------- User Inputs -----------

# Experience
experience = st.slider("Years of Coding Experience", 0, 50, 5)

# ML Experience
ml_experience = st.slider("Years of Machine Learning Experience", 0, 50, 2)

# Cloud/ML Spend
cloud_spend_input = st.selectbox("Money Spent on ML/Cloud in Last 5 Years ($USD)", [
    '$0 ($USD)', '$1-$99', '$100-$999', '$1000-$9,999', '$10,000-$99,999', '$100,000 or more ($USD)'
])
spend_map = {
    '$0 ($USD)': 0, '$1-$99': 50, '$100-$999': 550,
    '$1000-$9,999': 5000, '$10,000-$99,999': 50000, '$100,000 or more ($USD)': 100000
}
cloud_spend = spend_map[cloud_spend_input]

# Country
country = st.selectbox("Country You Reside In", [
    'United States of America', 'India', 'France', 'Germany', 'United Kingdom', 'Canada', 'Other'
])

# Role
role = st.selectbox("Current Role", [
    'Data Scientist', 'Data Analyst', 'ML Engineer', 'Research Scientist',
    'Software Engineer', 'Statistician', 'Other'
])

# Industry
industry = st.selectbox("Industry of Current Employer", [
    'Online Service/Internet-based Services', 'Academics/Education', 'Finance',
    'Medical/Pharmaceutical', 'Government/Public Service', 'Other'
])

# ML Maturity
ml_maturity_input = st.selectbox("Does Your Employer Use ML Methods?", [
    'No (we do not use ML methods)',
    'We are exploring ML methods (and may one day put a model into production)',
    'We use ML methods for generating insights (but do not put working models into production)',
    'We recently started using ML methods (i.e., models in production for less than 2 years)',
    'We have well established ML methods (i.e., models in production for more than 2 years)'
])
ml_maturity_map = {
    'No (we do not use ML methods)': 0,
    'We are exploring ML methods (and may one day put a model into production)': 1,
    'We use ML methods for generating insights (but do not put working models into production)': 2,
    'We recently started using ML methods (i.e., models in production for less than 2 years)': 3,
    'We have well established ML methods (i.e., models in production for more than 2 years)': 4
}
ml_maturity = ml_maturity_map[ml_maturity_input]

# Education
education_input = st.selectbox("Highest Education Level", [
    'No formal education past high school',
    'Some college/university study without earning a bachelor’s degree',
    'Bachelor’s degree',
    'Master’s degree',
    'Doctoral degree',
    'Professional doctorate'
])
education_order = {
    'No formal education past high school': 0,
    'Some college/university study without earning a bachelor’s degree': 1,
    'Bachelor’s degree': 2,
    'Master’s degree': 3,
    'Doctoral degree': 4,
    'Professional doctorate': 4.5
}
education_level = education_order[education_input]

# -------------- DataFrame Construction ----------------
input_dict = {
    'experience_years': experience,
    'ml_experience_years': ml_experience,
    'cloud_spend': cloud_spend,
    'education_level': education_level,
    'ml_maturity': ml_maturity
}

# Dummy column encoding
for col in feature_cols:
    if col.startswith("role_"):
        input_dict[col] = 1 if f"role_{role}" == col else 0
    elif col.startswith("country_"):
        input_dict[col] = 1 if f"country_{country}" == col else 0
    elif col.startswith("industry_"):
        input_dict[col] = 1 if f"industry_{industry}" == col else 0
    elif col.startswith(("Q124", "Q157", "Q179")):
        input_dict[col] = 0  # Assume no tool selected (can extend later)

input_df = pd.DataFrame([input_dict])

# -------------- Prediction ----------------
if st.button("Predict Salary"):
    try:
        # 🛠 Ensure all expected columns are present
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        salary = model.predict(input_df)[0]
        st.success(f"💰 Estimated Salary: ${int(salary):,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
