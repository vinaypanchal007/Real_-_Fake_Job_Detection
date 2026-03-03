# Real_-_Fake_Job_Predictor

## Hybrid Machine Learning System for Fraudulent Job Detection

A hybrid machine learning project to detect fraudulent job postings using NLP, structured metadata, and Logistic Regression.

This project demonstrates how text processing, structured feature engineering, class imbalance handling, model tuning, interpretability, and web deployment work together in a practical fraud detection system.

---

## Dataset Used

**Source:** Kaggle – *Real or Fake Job Posting Prediction*

### Labels

* **0 → Real Job**
* **1 → Fake Job**

The dataset contains only binary labels.
There is no separate “Unsure” category in the original data.

### Class Distribution

The dataset is highly imbalanced:

* ~95% Real jobs
* ~5% Fake jobs

Special techniques are applied to handle this imbalance.

---

# Model Architecture

This project uses a hybrid pipeline combining textual and structured features.

---

## 1. Text Features (NLP)

The following fields are combined into a single text column:

* Title
* Company Profile
* Description
* Requirements
* Benefits

### Text Processing

* TF-IDF Vectorization
* Unigrams + Bigrams
* Stopword removal
* Maximum 5000 features

---

## 2. Structured Metadata Features

### Categorical Features

* Location
* Department
* Employment Type
* Required Experience
* Required Education
* Industry
* Function

Encoding method:

* OneHotEncoder (handle_unknown='ignore')

### Numeric Features

* Extracted minimum salary from salary range
* Telecommuting (0/1)
* Has company logo (0/1)
* Has screening questions (0/1)

Processing method:

* StandardScaler

---

## 3. Class Imbalance Handling

Because the dataset is highly imbalanced:

* SMOTE (Synthetic Minority Oversampling Technique) is applied on training data only.

This improves fraud recall while maintaining strong overall performance.

---

## 4. Model

* Logistic Regression
* Hyperparameter tuning using GridSearchCV
* 5-fold cross-validation
* Evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC

---

# Model Performance (Hybrid Tuned Model)

* **Test Accuracy:** ≈ 0.98–0.99
* **Fraud Precision:** ≈ 0.85+
* **Fraud Recall:** ≈ 0.85+
* **ROC-AUC:** ≈ 0.98

The model maintains strong fraud detection performance while minimizing false positives.

---

# Decision Logic (Application Layer)

Although the dataset is binary (0 or 1), the deployed web application introduces a third category:

* **Fake Probability ≤ 0.30 → Real Job**
* **Fake Probability between 0.30 and 0.70 → Unsure**
* **Fake Probability ≥ 0.70 → Fake Job**

The “Unsure” category is introduced to reduce overconfident borderline predictions and improve reliability.

---

# Web Application

A Streamlit web application is built to:

* Accept job text details
* Accept structured job metadata
* Include credibility signals (logo, telecommuting, screening questions)
* Output classification as:

  * Real
  * Fake
  * Unsure

The raw probability score is hidden by default to prevent misuse or misinterpretation.

---

# Feature Insights

The model learned meaningful fraud indicators.

### Fraud-Associated Signals

* Words like “earn”, “money”, “data entry”
* Work-from-home patterns
* Certain metadata distributions

### Real-Job Indicators

* Corporate terms such as “team”, “clients”, “enterprise”
* Presence of company logo
* Screening questions

This confirms that both textual and structured features contribute meaningfully to fraud detection.

---

# Limitations

* The dataset contains only binary labels.
* Dataset bias may influence location or industry signals.
* Subtle, well-written scams may evade detection.
* Performance depends on how representative the training data is of real-world postings.

---

# Learning Outcomes

This project demonstrates:

* NLP feature engineering using TF-IDF
* Hybrid modeling (text + structured data)
* Handling class imbalance with SMOTE
* Hyperparameter tuning and cross-validation
* Model interpretability via feature importance
* Deployment using Streamlit

---

## Conclusion

This project showcases an end-to-end machine learning workflow: from data preprocessing and feature engineering to model optimization and real-world deployment.

It demonstrates how classical machine learning techniques can be effectively combined with structured metadata and thoughtful application-layer logic to build a practical fraud detection system.