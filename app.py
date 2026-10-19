
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from io import StringIO

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="SmartWastePrediction",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ SmartWastePrediction")
st.subheader(
    "Intelligent Municipal Solid Waste Generation Prediction Using Machine Learning"
)

st.write(
    "Select an area and enter its details to estimate daily waste generation."
)

# ---------------- LOAD DATASET ----------------
DATA_FILE = Path("waste_data.csv")
MODEL_FILE = Path("waste_model.pkl")

if not DATA_FILE.exists():
    st.error("waste_data.csv was not found in the project folder.")
    st.stop()

try:
    # Also supports files where each entire CSV row was accidentally quoted.
    raw_lines = [
        line.strip()
        for line in DATA_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    raw_lines = [
        line[1:-1]
        if line.startswith('"') and line.endswith('"')
        else line
        for line in raw_lines
    ]

    df = pd.read_csv(StringIO("\n".join(raw_lines)))
    df.columns = df.columns.str.strip().str.strip('"')

except Exception as error:
    st.error(f"Could not read the dataset: {error}")
    st.stop()

required_columns = [
    "date",
    "area",
    "population",
    "temperature",
    "previous_day_waste",
    "actual_waste"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing CSV columns: {missing_columns}")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

# Convert numeric columns
numeric_columns = [
    "population",
    "temperature",
    "previous_day_waste",
    "actual_waste"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["area"] = df["area"].astype("string").str.strip()

df = df.dropna(
    subset=["area", "population", "temperature",
            "previous_day_waste", "actual_waste"]
)

if df.empty:
    st.error("The dataset has no valid records.")
    st.stop()

# ---------------- DASHBOARD ----------------
st.success("Dataset loaded successfully.")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Number of Areas", df["area"].nunique())
col3.metric(
    "Average Waste",
    f"{df['actual_waste'].mean():,.1f} kg/day"
)

st.divider()

st.header("📊 Waste Generation Dashboard")

st.subheader("Area-wise Average Waste")

area_summary = (
    df.groupby("area", as_index=False)["actual_waste"]
    .mean()
    .sort_values("actual_waste", ascending=False)
)

st.bar_chart(
    area_summary.set_index("area")["actual_waste"],
    x_label="Area",
    y_label="Average waste (kg/day)"
)

st.subheader("Dataset Preview")
st.dataframe(df, use_container_width=True)

# ---------------- AREA FILTER ----------------
st.divider()
st.header("🔍 Explore an Area")

areas = sorted(df["area"].dropna().unique().tolist())

selected_dashboard_area = st.selectbox(
    "Choose an area to view its records",
    areas,
    key="dashboard_area"
)

filtered_df = df[df["area"] == selected_dashboard_area]

st.write(f"Records for **{selected_dashboard_area}**")
st.dataframe(filtered_df, use_container_width=True)

# ---------------- LOAD MODEL ----------------
st.divider()
st.header("🔮 Predict Waste Generation")

if not MODEL_FILE.exists():
    st.warning(
        "The trained model file was not found. "
        "Run `python train_model.py` in the VS Code terminal first."
    )
    st.stop()

try:
    saved_model = joblib.load(MODEL_FILE)

    # Supports the model dictionary created by train_model.py
    if isinstance(saved_model, dict):
        model = saved_model["model"]
        features = saved_model.get(
            "features",
            ["population", "temperature", "previous_day_waste"]
        )
    else:
        model = saved_model
        features = [
            "population",
            "temperature",
            "previous_day_waste"
        ]

except Exception as error:
    st.error(
        f"Could not load the model: {error}. "
        "Try retraining it using python train_model.py."
    )
    st.stop()

# ---------------- PREDICTION FORM ----------------
with st.form("prediction_form"):
    selected_area = st.selectbox(
        "Select Area for Prediction",
        areas,
        key="prediction_area"
    )

    area_records = df[df["area"] == selected_area]

    default_population = max(
        1, int(area_records["population"].median())
    )

    default_previous_waste = max(
        0.0, float(area_records["previous_day_waste"].median())
    )

    population = st.number_input(
        "Population",
        min_value=1,
        value=default_population,
        step=100
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=28.0,
        step=1.0
    )

    previous_day_waste = st.number_input(
        "Previous Day Waste (kg)",
        min_value=0.0,
        value=default_previous_waste,
        step=10.0
    )

    submitted = st.form_submit_button("Predict Waste")

if submitted:
    try:
        input_data = pd.DataFrame([{
            "population": population,
            "temperature": temperature,
            "previous_day_waste": previous_day_waste
        }])

        # Preserve the input order expected by the trained model.
        input_data = input_data[features]

        prediction = float(model.predict(input_data)[0])
        prediction = max(0.0, prediction)

        st.subheader(f"Prediction for {selected_area}")
        st.metric(
            "Estimated Daily Waste",
            f"{prediction:,.2f} kg"
        )

        st.info(
            "Use this estimate to demonstrate waste collection planning. "
            "It is not a verified municipal forecast."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption(
    "B.Tech CSE Major Project | SmartWastePrediction | "
    "Sample data is used for demonstration."
)
