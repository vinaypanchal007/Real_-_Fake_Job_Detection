import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
import joblib

frj = pd.read_csv("./Dataset/fake_real_job.csv")

frj = frj.drop(columns=['job_id'], errors='ignore')

text_cols = frj.select_dtypes(include='object').columns
text_cols = text_cols.drop(['salary_range','required_education','employment_type'], errors='ignore')

frj['combined_text'] = frj[text_cols].fillna("").agg(' '.join, axis=1)

frj['salary_range'] = frj['salary_range'].str.replace('$','',regex=False)
frj['salary_range'] = frj['salary_range'].str.replace(' per month','',regex=False)

split_salary = frj['salary_range'].str.split('-', expand=True)

frj['salary_min'] = pd.to_numeric(split_salary[0], errors='coerce').fillna(0).astype(int)
frj['salary_max'] = pd.to_numeric(split_salary[1], errors='coerce').fillna(frj['salary_min']).astype(int)

frj['exp_req(yrs)'] = frj['required_experience'].str.extract('(\d+)')
frj['exp_req(yrs)'] = frj['exp_req(yrs)'].fillna(0).astype(int)

frj = frj[[
    'combined_text',
    'employment_type',
    'required_education',
    'salary_min',
    'salary_max',
    'exp_req(yrs)',
    'telecommuting',
    'fraudulent'
]]

tr, te = train_test_split(frj, test_size=0.2, random_state=42)

x_tr = tr.drop(columns=['fraudulent'])
y_tr = tr['fraudulent']

x_te = te.drop(columns=['fraudulent'])
y_te = te['fraudulent']

text_col = 'combined_text'
cat_cols = ['employment_type','required_education']
num_cols = ['salary_min','salary_max','exp_req(yrs)','telecommuting']

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words='english',
                                 max_features=30000,
                                 ngram_range=(1,2),
                                 min_df=2), text_col),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MultinomialNB())
])

pipeline.fit(x_tr, y_tr)

y_pred = pipeline.predict(x_te)
y_pred_tr = pipeline.predict(x_tr)

print("Accuracy (Test):", accuracy_score(y_te, y_pred))
print("Accuracy (Train):", accuracy_score(y_tr, y_pred_tr))

print("\nClassification Report (Test):")
print(classification_report(y_te, y_pred))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_te, y_pred))

joblib.dump(pipeline, "./Joblib_Model/fakejob_pipeline.joblib")

print("Model saved successfully.")