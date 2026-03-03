import streamlit as st
import joblib
import re
import pandas as pd

st.set_page_config(page_title="Real & Fake Job Detector", layout="centered")

MODEL_PATH = "fakejob.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def decide_label(fake_prob):
    if fake_prob >= 0.7:
        return "Fake Job"
    elif fake_prob <= 0.3:
        return "Real Job"
    else:
        return "Unsure"

def is_gibberish(text):
    letters = re.findall(r"[a-zA-Z]", text)
    return len(letters) / max(len(text), 1) < 0.4

st.title("Real & Fake Job Posting Detection")

title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

location = st.text_input("Location")
department = st.text_input("Department")
employment_type = st.text_input("Employment Type")
required_experience = st.text_input("Required Experience")
required_education = st.text_input("Required Education")
industry = st.text_input("Industry")
function = st.text_input("Function")

salary_range = st.text_input("Salary Range (e.g. 50000-70000)")

telecommuting = st.selectbox("Telecommuting Available?", ["No", "Yes"])
has_company_logo = st.selectbox("Has Company Logo?", ["No", "Yes"])
has_questions = st.selectbox("Has Screening Questions?", ["No", "Yes"])

telecommuting = 1 if telecommuting == "Yes" else 0
has_company_logo = 1 if has_company_logo == "Yes" else 0
has_questions = 1 if has_questions == "Yes" else 0

if st.button("Predict"):

    combined_text = " ".join([
        title,
        description,
        requirements,
        company_profile,
        benefits
    ]).strip()

    if len(combined_text.split()) < 10:
        st.warning("Please provide more detailed job information.")
        st.stop()

    if is_gibberish(combined_text):
        st.error("Input text appears meaningless.")
        st.stop()

    salary_min = 0
    if salary_range:
        try:
            salary_min = float(salary_range.split("-")[0])
        except ValueError:
            salary_min = 0

    input_df = pd.DataFrame([{
        "combined_text": combined_text,
        "location": location if location else "Unknown",
        "department": department if department else "Unknown",
        "employment_type": employment_type if employment_type else "Unknown",
        "required_experience": required_experience if required_experience else "Unknown",
        "required_education": required_education if required_education else "Unknown",
        "industry": industry if industry else "Unknown",
        "function": function if function else "Unknown",
        "salary_min": salary_min,
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions
    }])

    fake_prob = model.predict_proba(input_df)[0][1]
    result = decide_label(fake_prob)

    st.markdown("### Prediction Result")

    if result == "Fake Job":
        st.error("FAKE JOB POSTING")
    elif result == "Real Job":
        st.success("REAL JOB POSTING")
    else:
        st.warning("UNSURE — NEEDS MANUAL REVIEW")

    st.caption("Prediction generated using model with text, metadata, salary, and credibility signals.")