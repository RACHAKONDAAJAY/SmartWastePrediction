
from pathlib import Path
from io import StringIO

import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Read and repair CSV formatting
text = Path("waste_data.csv").read_text(encoding="utf-8-sig")
lines = [line.strip() for line in text.splitlines() if line.strip()]

# Remove quotation marks around an entire row, if present
lines = [
    line[1:-1] if line.startswith('"') and line.endswith('"') else line
    for line in lines
]

df = pd.read_csv(StringIO("\n".join(lines)))
df.columns = df.columns.str.strip().str.strip('"')

required = [
    "population", "temperature",
    "previous_day_waste", "actual_waste"
]

if not all(column in df.columns for column in required):
    raise ValueError(
        f"Incorrect CSV columns: {df.columns.tolist()}"
    )

for column in required:
    df[column] = pd.to_numeric(df[column], errors="raise")

df = df.dropna(subset=required).reset_index(drop=True)

if len(df) < 10:
    raise ValueError("Please provide at least 10 valid dataset rows.")

print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())

# 2. Prepare inputs and target
features = ["population", "temperature", "previous_day_waste"]
X = df[features]
y = df["actual_waste"]

# 3. Train on earlier records, test on later records
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 4. Compare two ML models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

results = []
best_model = None
best_name = None
best_mae = float("inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE (kg)": round(mae, 2),
        "RMSE (kg)": round(rmse, 2),
        "R2 Score": round(r2, 3)
    })

    print(f"\n{name}")
    print("MAE:", round(mae, 2), "kg")
    print("RMSE:", round(rmse, 2), "kg")
    print("R2 Score:", round(r2, 3))

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name

# 5. Save the best model and metrics
joblib.dump(
    {"model": best_model, "features": features},
    "waste_model.pkl"
)

pd.DataFrame(results).to_csv("model_comparison.csv", index=False)

print("\nBest model:", best_name)
print("Saved model: waste_model.pkl")
print("Saved comparison: model_comparison.csv")
print("\nActual vs predicted:")
predictions = best_model.predict(X_test)

for actual, predicted in zip(y_test, predictions):
    print(f"Actual: {actual:.1f} kg | Predicted: {predicted:.1f} kg")
