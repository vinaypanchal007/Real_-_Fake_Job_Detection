# Fake Job Posting Detector

## Project Overview

The **Fake Job Posting Detector** is a Machine Learning web application that predicts whether a job posting is **real or fraudulent**.

Many online job portals contain scam postings designed to collect personal information or scam job seekers. This project uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze job details and detect suspicious postings.

The model analyzes the **job description, requirements, company information, salary, experience, and other job attributes** to determine the authenticity of the posting.

The application is deployed using **Streamlit**, allowing users to enter job details through a simple web interface and instantly receive a prediction.

---

## Features

* Detects **Real vs Fake job postings**
* Uses **Natural Language Processing (TF-IDF)** for text analysis
* Combines **text features, categorical features, and numerical features**
* Provides **probability-based prediction**
* User-friendly **Streamlit interface**
* Handles incomplete or low-information job postings with validation

---

## Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Streamlit**
* **Joblib**

---

## Machine Learning Pipeline

The model follows a structured ML pipeline:

### 1. Data Preprocessing

* Combine multiple text fields into a single **combined_text** column
* Extract **salary_min** and **salary_max** from salary range
* Extract **experience (years)** from experience column
* Remove unnecessary columns

### 2. Feature Engineering

Three types of features are used:

**Text Features**

* TF-IDF Vectorization
* Unigrams and Bigrams
* Max Features: 30,000

**Categorical Features**

* Employment Type
* Required Education
* Encoded using **OneHotEncoder**

**Numerical Features**

* Minimum Salary
* Maximum Salary
* Experience Required
* Telecommuting

### 3. Model

The final classifier used is:

**Multinomial Naive Bayes**

The preprocessing and model are combined into a single **Scikit-learn Pipeline**.

---

## Model Performance

Example performance on test data:

* **Accuracy:** ~95%
* **Precision:** ~0.95
* **Recall:** ~0.95
* **F1 Score:** ~0.95

The model performs well in identifying fraudulent job postings.

---

## Project Structure

```
Fake-Job-Detector
│
├── Dataset
│   └── fake_real_job.csv
│
├── Joblib_Model
│   └── fakejob_pipeline.joblib
│
├── train_model.py
├── streamlit.py
├── requirements.txt
└── README.md

##