# 💼 Data Scientist Salary Predictor

This project is part of a university data analytics case study using the Kaggle 2022 survey data. It allows users to input their experience, background, and job-related details to predict their estimated salary in the data science profession.

---

## 🔍 Overview

We trained a machine learning model using responses from over 25,000 data professionals worldwide, leveraging features like:

- Coding and ML experience
- Education level
- Industry and job role
- Country of residence
- Cloud/ML spending history
- ML maturity at the workplace

The model is deployed via a Streamlit app and allows students or professionals to project their potential income based on selected characteristics.

---

## 🧠 Model

- **Algorithm:** RandomForestRegressor
- **Target Variable:** Salary (mapped from Q29 Kaggle survey salary ranges)
- **Features:** Categorical (one-hot encoded) and numerical features derived from:
  - Q4 (Country)
  - Q8 (Education)
  - Q11 (Programming Experience)
  - Q16 (ML Experience)
  - Q24 (Industry)
  - Q27 (ML Maturity)
  - Q30 (Cloud Spend)
  - Q23 (Job Role)
  - Q124/Q157/Q179 (Tools used)

---

## 🚀 Run the App Locally

```bash
git clone https://github.com/your-username/salary-predictor-app.git
cd salary-predictor-app
pip install -r requirements.txt
streamlit run app.py
