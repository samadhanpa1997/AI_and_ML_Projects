import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("loan_data.csv")

print("\nFirst 5 Records")
print(df.head())

print("\nShape")
print(df.shape)

print("\nInfo")
print(df.info())

print("\nDescribe")
print(df.describe())

# ==========================
# Missing Values
# ==========================

print(df.isnull().sum())

df.fillna(df.mode().iloc[0], inplace=True)

# ==========================
# Remove Loan_ID
# ==========================

if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# ==========================
# Encoding
# ==========================

encoder = LabelEncoder()

cat_cols = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

df["Dependents"] = df["Dependents"].replace("3+",3)
df["Dependents"] = df["Dependents"].astype(int)

# ==========================
# X and Y
# ==========================

X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]

# ==========================
# Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Models
# ==========================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

os.makedirs("models", exist_ok=True)

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print("\n", "="*50)
    print(name)
    print("="*50)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("Confusion Matrix")
    print(confusion_matrix(y_test,y_pred))

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    file_name = name.lower().replace(" ", "_")

    with open(
        f"models/{file_name}_model.pkl",
        "wb"
    ) as file:
        pickle.dump(model, file)

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nModel Comparison")
print(results_df)