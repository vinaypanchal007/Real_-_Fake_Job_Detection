import streamlit as st
import joblib
import pandas as pd

MODEL_PATH = "./fakejob_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Fake Job Detector", layout="centered")

st.title("Fake Job Posting Detector")
st.write("Enter job details to check if the job posting is Real or Fake.")

title = st.text_input("Job Title")

company_name = st.text_input("Company Name")

description = st.text_area("Job Description")

requirements = st.text_area("Requirements")

benefits = st.text_area("Benefits")

location = st.text_input("Location")

industry = st.text_input("Industry")

department = st.text_input("Department")

employment_type = st.selectbox(
    "Employment Type",
    [
        "Full-time",
        "Part-time",
        "Contract",
        "Temporary"
    ]
)

required_education = st.selectbox(
    "Required Education",
    [
        "High School",
        "Associate Degree",
        "Bachelor's Degree",
        "Master's Degree",
        "Diploma"
    ]
)

salary_min = st.number_input("Minimum Salary", min_value=0)

salary_max = st.number_input("Maximum Salary", min_value=0)

experience = st.number_input("Required Experience (Years)", min_value=0)

telecommuting = st.selectbox("Telecommuting?", ["No", "Yes"])

if telecommuting == "Yes":
    telecommuting = 1
else:
    telecommuting = 0


if st.button("Predict Job Authenticity"):

    combined_text = " ".join([
        title,
        company_name,
        description,
        requirements,
        benefits,
        location,
        department,
        industry
    ])

    if combined_text.strip() == "":
        st.warning("Please enter job details before predicting.")
        st.stop()

    if len(combined_text.split()) < 10:
        st.warning("Please provide more detailed job information.")
        st.stop()

    salary_min = int(salary_min)
    salary_max = int(salary_max)
    experience = int(experience)

    if salary_max <= salary_min:
        st.warning("Maximum salary should be greater than minimum salary.")
        st.stop()

    input_df = pd.DataFrame({
        "combined_text": [combined_text],
        "employment_type": [employment_type],
        "required_education": [required_education],
        "salary_min": [salary_min],
        "salary_max": [salary_max],
        "exp_req(yrs)": [experience],
        "telecommuting": [telecommuting]
    })

    fake_probability = model.predict_proba(input_df)[0][1]

    if fake_probability > 0.6:
        st.error("Fake Job Posting")

    elif fake_probability < 0.4:
        st.success("Real Job Posting")

    else:

        st.warning("Uncertain Prediction")
