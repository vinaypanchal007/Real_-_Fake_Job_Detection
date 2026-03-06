import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
import joblib

frj = pd.read_csv(r"./Dataset/fake_real_job.csv")

frj = frj.drop(columns=['job_id'], errors='ignore')

text_cols = frj.select_dtypes(include='object').columns
text_cols = text_cols.drop(['salary_range', 'required_education','employment_type'], errors='ignore')
print(text_cols)

hotenc_cols = frj[['employment_type', 'required_education']].columns
print(hotenc_cols)

frj['combined_text'] = frj[text_cols].agg(' '.join, axis=1)

frj['salary_range'] = frj['salary_range'].str.replace('$','',regex=False)
frj['salary_range'] = frj['salary_range'].str.replace(' per month','',regex=False)

split_salary = frj['salary_range'].str.split('-', expand=True)

frj['salary_min'] = pd.to_numeric(split_salary[0], errors='coerce')
frj['salary_max'] = pd.to_numeric(split_salary[1], errors='coerce').fillna(frj['salary_min']).astype(int)

frj['exp_req(yrs)'] = frj['required_experience'].str.extract('(\d+)')
frj['exp_req(yrs)'] = frj['exp_req(yrs)'].fillna(0)
frj['exp_req(yrs)'] = frj['exp_req(yrs)'].astype(int)

print(frj.head())

for i in text_cols:
    frj = frj.drop(columns=[i], errors='ignore')

frj = frj.drop(columns=['salary_range', 'required_experience'], errors='ignore')
print(frj.head())

tr, te = train_test_split(frj, test_size=0.2, random_state=42)

x_tr = tr.drop(columns=['fraudulent'], errors='ignore')
y_tr = tr['fraudulent']
x_te = te.drop(columns=['fraudulent'], errors='ignore')
y_te = te['fraudulent']

num_cols = frj[['salary_min', 'salary_max', 'exp_req(yrs)', 'telecommuting']].columns

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words='english', max_features=30000, ngram_range=(1, 2), min_df=2), 'combined_text'),
        ('hotenc', OneHotEncoder(handle_unknown='ignore'), hotenc_cols),
        ('num', 'passthrough', num_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MultinomialNB())
])

pipeline.fit(x_tr, y_tr)

y_pred = pipeline.predict(x_te)
y_pred_tr = pipeline.predict(x_tr)

print("Accuracy:", accuracy_score(y_te, y_pred))
print("Accuracy:", accuracy_score(y_tr, y_pred_tr))

print("Classification Report (Test):\n", classification_report(y_te, y_pred))
print("Confusion Matrix (Test):\n", confusion_matrix(y_te, y_pred))

print("Classification Report (Train):\n", classification_report(y_tr, y_pred_tr))
print("Confusion Matrix (Train):\n", confusion_matrix(y_tr, y_pred_tr))

joblib.dump(pipeline, "./Joblib_Model/fakejob_pipeline.joblib")
print("Model saved successfully.")