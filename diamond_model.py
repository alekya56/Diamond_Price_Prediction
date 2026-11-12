
# ============================================
# DIAMOND PRICE PREDICTION
# COMPARISON OF FOUR ML ALGORITHMS
# ============================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("diamonds.csv")

print("\nFIRST 5 ROWS")
print(df.head())

print("\nORIGINAL DATASET SHAPE:", df.shape)

# ============================================
# 2. DATA CLEANING
# ============================================

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Remove duplicate records
df = df.drop_duplicates()

# Remove invalid dimensions
df = df[
    (df["x"] > 0) & (df["x"] <= 20) &
    (df["y"] > 0) & (df["y"] <= 20) &
    (df["z"] > 0) & (df["z"] <= 20)
]

df = df.dropna()

print("\nCLEANED DATASET SHAPE:", df.shape)

# ============================================
# 3. FEATURES AND TARGET
# ============================================

X = df.drop("price", axis=1)
y = df["price"]

categorical_features = [
    "cut", "color", "clarity"
]

numerical_features = [
    "carat", "depth", "table", "x", "y", "z"
]

# ============================================
# 4. PREPROCESSING
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# ============================================
# 5. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)

# ============================================
# 6. DEFINE FOUR ML MODELS
# ============================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "KNN Regressor": KNeighborsRegressor(
        n_neighbors=5,
        n_jobs=-1
    )
}

# ============================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================

results = []
trained_models = {}

for name, model in models.items():

    print("\n================================")
    print("TRAINING:", name)
    print("================================")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

    trained_models[name] = pipeline

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# ============================================
# 8. MODEL COMPARISON TABLE
# ============================================

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n\n========================================")
print("          MODEL COMPARISON")
print("========================================")

print(comparison_df.to_string(index=False))

# ============================================
# 9. FIND BEST MODEL
# ============================================

best_model_name = comparison_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n========================================")
print("BEST PERFORMING MODEL")
print("========================================")

print("Model:", best_model_name)

print(
    "R2 Score:",
    comparison_df.iloc[0]["R2 Score"]
)

# ============================================
# 10. SAVE COMPARISON RESULTS
# ============================================

comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)

# ============================================
# 11. SAVE BEST MODEL
# ============================================

joblib.dump(
    best_model,
    "diamond_price_model.pkl"
)

print("\nComparison table saved successfully!")

print("Best model saved as diamond_price_model.pkl")

print("\nALL FOUR MODELS TRAINED SUCCESSFULLY!")



# ============================================
# 12. SAVE KNN MODEL USING PICKLE
# ============================================

import pickle
import os

os.makedirs("models", exist_ok=True)

with open("models/diamond_knn_pipeline.pkl", "wb") as file:
    pickle.dump(
        trained_models["KNN Regressor"],
        file
    )

print("KNN pipeline saved successfully!")
