#%%writefile train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Create necessary directories
os.makedirs("data", exist_ok=True)

# 1. Load model and metadata instead of retraining if model.pkl already exists
if os.path.exists("model.pkl"):
    print("--- Model and metadata found! Loading from disk ---")
    saved_package = joblib.load("model.pkl")
    model = saved_package["model"]
    metadata = saved_package["metadata"]
    print(f"Loaded model trained on: {metadata['training_date']}")
else:
    print("--- Training new Logistic Regression Pipeline & generating metadata ---")
    np.random.seed(42)
    n_samples = 2000

    # Generate dataset
    data = {
        "income": np.random.exponential(scale=50000, size=n_samples) + 20000,
        "loan_amount": np.random.exponential(scale=15000, size=n_samples) + 5000,
        "credit_score": np.random.randint(550, 850, size=n_samples),
        "employment_length": np.random.randint(0, 20, size=n_samples),
        "loan_intent": np.random.choice(
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"], size=n_samples
        ),
    }
    df = pd.DataFrame(data)

    # Introduce realistic default probability logic
    default_prob = (
        (df["loan_amount"] / df["income"] > 0.4).astype(int) * 0.4
        + (df["credit_score"] < 620).astype(int) * 0.5
        + np.random.normal(0, 0.1, n_samples)
    )
    df["default"] = (default_prob > 0.35).astype(int)

    # Add missing values to test robust imputation in pipeline
    df.loc[np.random.choice(df.index, 50), "credit_score"] = np.nan

    # Save raw dataset
    df.to_csv("data/loan_data.csv", index=False)

    # 2. Define Features (X) and Target (y)
    X = df.drop(columns=["default"])
    y = df["default"]

    numeric_features = ["income", "loan_amount", "credit_score", "employment_length"]
    categorical_features = ["loan_intent"]

    # 3. Build Preprocessing & Modeling Pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Mandatory Logistic Regression classifier[cite: 4]
    classifier = LogisticRegression(random_state=42, max_iter=1000)

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    # 4. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- Fitting Pipeline ---")
    model.fit(X_train, y_train)

    # 5. Evaluation metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # 6. Define Metadata Dictionary
    metadata = {
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "Logistic Regression with ColumnTransformer Pipeline",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "default",
        "classes": [0, 1],
        "metrics": {
            "roc_auc": float(roc_auc)
        },
        "allowed_loan_intents": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"]
    }

    # Save model pipeline and metadata together using joblib
    saved_package = {
        "model": model,
        "metadata": metadata
    }
    joblib.dump(saved_package, "model.pkl")  # Save after training
    print("\n✅ Model and Metadata successfully saved to model.pkl")
