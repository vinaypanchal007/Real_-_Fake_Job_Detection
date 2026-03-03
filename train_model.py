import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

df = pd.read_csv("fake_job_postings.csv")
df['fraudulent'] = df['fraudulent'].astype(int)

text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
cat_cols = [
    'location','department','employment_type',
    'required_experience','required_education',
    'industry','function'
]
binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']

df[text_cols] = df[text_cols].fillna('')
df[cat_cols] = df[cat_cols].fillna('Unknown')
df[binary_cols] = df[binary_cols].fillna(0)

df['combined_text'] = (
    df['title'] + " " +
    df['description'] + " " +
    df['requirements'] + " " +
    df['company_profile'] + " " +
    df['benefits']
)

df['salary_min'] = df['salary_range'].str.split('-').str[0]
df['salary_min'] = pd.to_numeric(df['salary_min'], errors='coerce').fillna(0)

num_cols = ['salary_min'] + binary_cols

X = df[['combined_text'] + cat_cols + num_cols]
y = df['fraudulent']

x_tr, x_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            stop_words='english',
            min_df=2
        ), 'combined_text'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ],
    remainder='drop'
)

pipeline = ImbPipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=2000, random_state=42))
])

param_grid = {
    'clf__C': [0.1, 0.5, 1, 2, 5]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid.fit(x_tr, y_tr)

print("Best C:", grid.best_params_)
print("Best CV F1 Score:", grid.best_score_)

best_model = grid.best_estimator_

y_train_pred = best_model.predict(x_tr)
y_test_pred = best_model.predict(x_te)

train_probs = best_model.predict_proba(x_tr)[:,1]
test_probs = best_model.predict_proba(x_te)[:,1]

print("\n----- TRAIN PERFORMANCE -----")
print(classification_report(y_tr, y_train_pred))
print("Train ROC-AUC:", roc_auc_score(y_tr, train_probs))

print("\n----- TEST PERFORMANCE -----")
print(classification_report(y_te, y_test_pred))
print("Test ROC-AUC:", roc_auc_score(y_te, test_probs))

print("\nConfusion Matrix:")
print(confusion_matrix(y_te, y_test_pred))

clean_pipeline = Pipeline([
    ('preprocess', best_model.named_steps['preprocess']),
    ('clf', best_model.named_steps['clf'])
])

joblib.dump(clean_pipeline, "fakejob.joblib")
print("\nDeployment-ready model saved successfully")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_te, test_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.show()