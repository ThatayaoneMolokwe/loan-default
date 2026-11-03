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

# Create directories
os.makedirs("data", exist_ok=True)

# Load instead of retraining if model.pkl already exists
if os.path.exists("model.pkl"):
    print("--- Model found! Loading instead of retraining ---")
    model = joblib.load("model.pkl")  # Load instead of retraining
else:
    print("--- Training new model ---")
    np.random.seed(42)
    n_samples = 2000

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

    print("--- Fitting Logistic Regression Pipeline ---")
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Save after training
    import joblib
    joblib.dump(model, "model.pkl")  # Save after training
    print("\n✅ Model successfully saved to model.pkl")
