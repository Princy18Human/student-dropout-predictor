import streamlit as st
import numpy as np
import pandas as pd
import joblib

# App configuration
st.set_page_config(page_title="🎓 Student Dropout & GPA Predictor", page_icon="🎯", layout="centered")

# Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(to right, #4facfe, #00f2fe);
        border: none;
        border-radius:8px;
        padding:10px 24px;
        font-size:16px;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #43e97b, #38f9d7);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎓 Student Dropout Risk & GPA Prediction")
st.write(
    "Use the sliders below to enter student data and estimate **dropout risk** and **predicted GPA**."
)

# Load models and feature names
clf = joblib.load("clf_model.pkl")
reg = joblib.load("reg_model.pkl")
feature_names = joblib.load("clf_features.pkl")

# User input
with st.form("input_form"):
    absences = st.slider("Absences", 0, 100, 5)
    failures = st.slider("Failures", 0, 4, 0)
    studytime = st.slider("Study Time (1=low, 4=high)", 1, 4, 2)
    G1 = st.slider("G1 Grade", 0, 20, 10)
    G2 = st.slider("G2 Grade", 0, 20, 10)
    G3 = st.slider("G3 Grade", 0, 20, 10)

    # Engineered features
    attendance_rate = 1 - (absences / 100)
    grade_improvement = G3 - G1

    # Submit button
    submitted = st.form_submit_button("💡 Predict Risk & GPA")

if submitted:
    # Input dictionary
    input_data = {
        "absences": absences,
        "failures": failures,
        "studytime": studytime,
        "G1": G1,
        "G2": G2,
        "G3": G3,
        "attendance_rate": attendance_rate,
        "grade_improvement": grade_improvement
    }

    input_df = pd.DataFrame([input_data])

    # Ensure all expected features exist
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[feature_names]

    # Predictions
    dropout_pred = clf.predict(input_df)[0]
    gpa_pred = reg.predict(input_df)[0]
    dropout_proba = clf.predict_proba(input_df)[0]

    # Animation
    st.balloons()

    # Risk text
    risk_text = "✅ No" if dropout_pred == 0 else "⚠️ Yes"

    # Result Expander (safe formatting)
    html_result = """
    <div style="background-color:#e6f7ff; padding:20px; border-radius:10px;">
        <h3 style="color:#007acc;">Predicted Dropout Risk: {}</h3>
        <h3 style="color:#007acc;">Predicted GPA: {:.2f}</h3>
        <p><b>Confidence:</b></p>
        <ul>
            <li>No Dropout: {:.1f}%</li>
            <li>Dropout: {:.1f}%</li>
        </ul>
    </div>
    """.format(
        risk_text,
        gpa_pred,
        dropout_proba[0]*100,
        dropout_proba[1]*100
    )

    with st.expander("🎯 Prediction Results", expanded=True):
        st.markdown(html_result, unsafe_allow_html=True)

    # Celebrate high GPA
    if gpa_pred >= 15:
        st.success("🎉 Excellent predicted GPA! Keep up the good work!")
    elif gpa_pred < 8:
        st.warning("⚠️ Low predicted GPA—consider additional support and intervention.")
