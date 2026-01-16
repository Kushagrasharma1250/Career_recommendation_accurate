import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

model = joblib.load("model.pkl")
mlb = joblib.load("mlb.pkl")  # Load the MultiLabelBinarizer
le = joblib.load("le.pkl")    # Load the LabelEncoder

st.title("Career Recommendation System")
education = st.selectbox(
    "Select your highest education level",
    ["Associate", "Diploma", "Bachelor", "Master", "PhD"]
)

skills = st.text_input("Enter your skills (comma separated)")

if st.button("Predict"):
    skill_list = [s.strip().lower() for s in skills.split(",")]
    skill_encoded = mlb.transform([skill_list])
    expected_skill_features = len(mlb.classes_)
    if skill_encoded.shape[1] < expected_skill_features:
        padding = np.zeros((skill_encoded.shape[0], expected_skill_features - skill_encoded.shape[1]))
        skill_encoded = np.hstack([skill_encoded, padding])
    edu_encoded = le.transform([education]).reshape(-1, 1)
    X = np.hstack([skill_encoded, edu_encoded])
    prediction = model.predict(X)
    st.success(f"Recommended career path: {prediction[0]}")