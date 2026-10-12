import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("student_depression_cleaned.csv")

num_cols = ["age", "academic_pressure", "cgpa", "study_satisfaction",
            "work_study_hours", "financial_stress"]
cat_cols = ["gender", "sleep_duration", "dietary_habits",
            "family_history_of_mental_illness"]

X = df[num_cols + cat_cols]
y = df["depression"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough")

model = Pipeline([
    ("pre", pre),
    ("rf", RandomForestClassifier(n_estimators=150, max_depth=12,
                                  min_samples_leaf=5, random_state=42, n_jobs=-1))
])
model.fit(X_train, y_train)
print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))

joblib.dump(model, "model.joblib", compress=3)
print("Saved model.joblib")