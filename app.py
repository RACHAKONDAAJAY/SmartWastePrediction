
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from io import StringIO

# ==========================================================
# INTELLIGENT MUNICIPAL SOLID WASTE PREDICTION
# ==========================================================

st.set_page_config(
    page_title="Intelligent Municipal Solid Waste Prediction",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = Path("waste_data.csv")
MODEL_FILE = Path("waste_model.pkl")
COMPARISON_FILE = Path("model_comparison.csv")

REQUIRED_COLUMNS = [
    "date", "area", "population", "temperature",
    "previous_day_waste", "actual_waste"
]

FEATURES = [
    "population", "temperature", "previous_day_waste"
]

AREA_GROUPS = {
    "Hyderabad / GHMC locality list": [
        "Ameerpet", "Alwal", "Bachupally", "Begumpet",
        "Chandanagar", "Charminar", "Dilsukhnagar",
        "Financial District", "Gachibowli", "Hayathnagar",
        "HITEC City", "Jubilee Hills", "Kapra", "Kondapur",
        "Kompally", "Kukatpally", "LB Nagar", "Madhapur",
        "Mehdipatnam", "Medchal", "Miyapur", "Malkajgiri",
        "Nanakramguda", "Nizampet", "Qutbullapur",
        "Secunderabad", "Serilingampally", "Shamshabad",
        "Uppal", "Hyderabad"
    ],
    "Nearby town": ["Choutuppal"]
}

ALL_AREAS = sorted({
    area
    for group in AREA_GROUPS.values()
    for area in group
})

# ==========================================================
# ENVIRONMENTAL DESIGN
# ==========================================================

st.markdown("""
<style>
.stApp {
    background-color: #F1F7F2;
    color: #20352A;
}
[data-testid="stHeader"] {
    background-color: #F1F7F2;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #175B3A !important;
    font-weight: 750;
}
p, label, .stMarkdown {
    color: #263D30;
}
[data-testid="stSidebar"] {
    background-color: #E0EEE3;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #C8DFCD;
    border-radius: 14px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    color: #176B45;
}
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background-color: #218653;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #155C39;
    color: white;
}
hr {
    border-color: #C8DFCD;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():
    if not DATA_FILE.exists():
        return None, "waste_data.csv was not found."

    try:
        raw = DATA_FILE.read_text(encoding="utf-8-sig")
        lines = [
            line.strip()
            for line in raw.splitlines()
            if line.strip()
        ]

        cleaned = []
        for line in lines:
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            cleaned.append(line)

        data = pd.read_csv(StringIO("\n".join(cleaned)))
        data.columns = [
            str(c).strip().lower().strip('"')
            for c in data.columns
        ]

        missing = [
            c for c in REQUIRED_COLUMNS if c not in data.columns
        ]
        if missing:
            return None, (
                f"Missing CSV columns: {missing}. "
                f"Found: {data.columns.tolist()}"
            )

        data["area"] = data["area"].astype("string").str.strip()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

        for col in [
            "population", "temperature",
            "previous_day_waste", "actual_waste"
        ]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=[
            "area", "population", "temperature",
            "previous_day_waste", "actual_waste"
        ])
        data = data[data["area"] != ""]

        if data.empty:
            return None, "The CSV contains no valid records."

        return data, None

    except Exception as exc:
        return None, f"Dataset error: {exc}"


df, data_error = load_dataset()

# ==========================================================
# LOAD EXISTING TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None, None, "waste_model.pkl was not found."

    try:
        saved = joblib.load(MODEL_FILE)

        if isinstance(saved, dict):
            model = saved.get("model")
            features = saved.get("features", FEATURES)
        else:
            model = saved
            features = FEATURES

        if model is None:
            return None, None, "The model file is invalid."

        return model, list(features), None

    except Exception as exc:
        return None, None, str(exc)


model, model_features, model_error = load_model()

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def csv_bytes(data):
    return data.to_csv(index=False).encode("utf-8")


def area_records(area):
    return df[df["area"].str.casefold() == area.casefold()]


def median_value(area, column, fallback):
    records = area_records(area)
    if not records.empty:
        value = records[column].median()
        if pd.notna(value):
            return float(value)

    value = df[column].median()
    return float(value) if pd.notna(value) else float(fallback)


def predict_waste(population, temperature, previous_waste):
    if model is None:
        raise ValueError("Trained ML model is unavailable.")

    values = {
        "population": population,
        "temperature": temperature,
        "previous_day_waste": previous_waste
    }

    missing = [
        f for f in model_features if f not in values
    ]
    if missing:
        raise ValueError(f"Unsupported model inputs: {missing}")

    inputs = pd.DataFrame([{
        f: values[f] for f in model_features
    }])
    return max(0.0, float(model.predict(inputs)[0]))


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.title("♻️ Waste Prediction")
    st.caption("Smart Environmental Management")
    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Waste Prediction",
            "Monthly Trends",
            "Waste Categories",
            "Collection Planning",
            "Area Explorer",
            "Reports & Downloads",
            "Model Performance",
            "Customer Care",
            "Help Center"
        ]
    )

    st.divider()
    st.caption("B.Tech CSE Major Project")

# ==========================================================
# HEADER
# ==========================================================

st.title("♻️ Intelligent Municipal Solid Waste Prediction")

st.markdown("""
### Smart Waste Management Using Machine Learning

**Predict. Monitor. Manage. Build a Cleaner Tomorrow.**

Analyze historical waste records, estimate waste generation,
compare localities, and plan more efficient waste collection.
""")

st.caption(
    "B.Tech CSE | Machine Learning | Waste Analytics | Sustainability"
)

if data_error:
    st.error(data_error)
    st.info(
        "Place waste_data.csv beside app_full.py and ensure it has "
        "the six required column names."
    )
    st.code(
        "date,area,population,temperature,"
        "previous_day_waste,actual_waste"
    )
    st.stop()

dataset_areas = sorted(df["area"].dropna().unique().tolist())
available_areas = sorted(set(dataset_areas + ALL_AREAS))

# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":
    st.subheader("📊 Waste Management Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recorded Areas", df["area"].nunique())
    col2.metric("Data Records", len(df))
    col3.metric(
        "Average Waste",
        f"{df['actual_waste'].mean():,.1f} kg"
    )
    col4.metric(
        "Highest Recorded Waste",
        f"{df['actual_waste'].max():,.1f} kg"
    )

    st.info(
        "Statistics describe your current CSV. If its records are "
        "generated demonstration data, these figures are not official."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Average Waste by Area")
        area_avg = (
            df.groupby("area")["actual_waste"]
            .mean().sort_values(ascending=False)
        )
        st.bar_chart(area_avg)

    with right:
        st.subheader("Historical Waste Trend")
        dated = df.dropna(subset=["date"])
        if not dated.empty:
            trend = (
                dated.groupby("date")["actual_waste"]
                .mean().sort_index()
            )
            st.line_chart(trend)
        else:
            st.warning("Valid dates are not available.")

    st.subheader("Recent Dataset Records")
    st.dataframe(
        df.sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Dataset CSV",
        csv_bytes(df),
        "waste_dataset.csv",
        "text/csv"
    )

# ==========================================================
# WASTE PREDICTION
# ==========================================================

elif page == "Waste Prediction":
    st.subheader("🔮 Machine-Learning Waste Prediction")

    if model is None:
        st.error("The existing model could not be loaded.")
        st.write(model_error)
        st.code("python train_model.py")
    else:
        st.success("Existing ML model loaded successfully.")

        with st.form("prediction_form"):
            area = st.selectbox("Select locality", available_areas)

            records = area_records(area)
            if records.empty:
                st.warning(
                    "No historical records for this area. Default "
                    "values use general dataset medians, not verified "
                    "population or temperature for this locality."
                )

            population = st.number_input(
                "Population",
                min_value=1,
                value=max(
                    1, int(median_value(area, "population", 1000))
                ),
                step=100
            )

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-20.0,
                max_value=60.0,
                value=28.0,
                step=1.0
            )

            previous = st.number_input(
                "Previous-day waste (kg)",
                min_value=0.0,
                value=max(
                    0.0,
                    median_value(area, "previous_day_waste", 500)
                ),
                step=10.0
            )

            submit = st.form_submit_button(
                "Predict Waste Generation",
                use_container_width=True
            )

        if submit:
            try:
                predicted = predict_waste(
                    population, temperature, previous
                )

                st.metric(
                    "Predicted Daily Waste",
                    f"{predicted:,.2f} kg/day"
                )

                comparison = pd.DataFrame({
                    "Measurement": [
                        "Previous-day waste",
                        "Predicted waste"
                    ],
                    "Waste (kg)": [previous, predicted]
                })

                if not records.empty:
                    actual = float(records["actual_waste"].iloc[-1])
                    comparison.loc[len(comparison)] = [
                        "Last available recorded actual",
                        actual
                    ]
                    st.metric(
                        "Last available recorded actual",
                        f"{actual:,.2f} kg"
                    )
                    st.metric(
                        "Prediction minus recorded actual",
                        f"{predicted - actual:,.2f} kg"
                    )
                else:
                    st.caption(
                        "No recorded actual waste is available for "
                        "this locality, so an actual-versus-predicted "
                        "difference cannot be calculated."
                    )

                st.bar_chart(
                    comparison.set_index("Measurement")
                )
                st.dataframe(comparison, hide_index=True)

                result = pd.DataFrame([{
                    "area": area,
                    "population_input": population,
                    "temperature_c": temperature,
                    "previous_day_waste_kg": previous,
                    "predicted_waste_kg_day": predicted,
                    "data_status": (
                        "Area records available"
                        if not records.empty
                        else "No historical records for this area"
                    ),
                    "result_type": "ML estimate, not official forecast"
                }])

                st.download_button(
                    "Download Prediction Report",
                    csv_bytes(result),
                    "waste_prediction.csv",
                    "text/csv"
                )

            except Exception as exc:
                st.error(f"Prediction error: {exc}")

# ==========================================================
# MONTHLY TRENDS
# ==========================================================

elif page == "Monthly Trends":
    st.subheader("📅 Historical Monthly Waste Analysis")

    dated = df.dropna(subset=["date"]).copy()

    if dated.empty:
        st.warning("No valid dates exist in your dataset.")
    else:
        dated["month"] = dated["date"].dt.to_period("M").astype(str)

        monthly = (
            dated.groupby("month")["actual_waste"]
            .mean().sort_index()
        )

        st.caption(
            "Monthly averages are calculated from available CSV records. "
            "Missing months are not invented."
        )

        st.line_chart(monthly)
        st.bar_chart(monthly)

        st.dataframe(
            monthly.rename("Average recorded waste (kg)")
            .reset_index(),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Monthly Report",
            csv_bytes(monthly.rename("average_waste_kg").reset_index()),
            "monthly_waste_report.csv",
            "text/csv"
        )

# ==========================================================
# WASTE CATEGORY ANALYSIS
# ==========================================================

elif page == "Waste Categories":
    st.subheader("🗑️ Four-Bin Waste Segregation Guide")

    st.warning(
        "Your existing dataset does not measure waste categories. "
        "The category breakdown below is illustrative, not measured "
        "municipal data."
    )

    st.markdown("""
    - 🟢 **Green bin — Wet / biodegradable waste:** food scraps,
      vegetable peels, and garden waste.
    - 🔵 **Blue bin — Dry recyclable waste:** clean paper, cardboard,
      glass, and accepted recyclable packaging.
    - 🔴 **Red bin — Sanitary / domestic hazardous waste:** follow
      local collection rules for sanitary and hazardous household items.
    - 🟡 **Yellow bin — Special-care waste:** certain batteries,
      electronic items, bulbs, and other items needing designated
      collection. Follow local authority guidance.
    """)

    area = st.selectbox("Select area", dataset_areas)
    total = float(area_records(area)["actual_waste"].mean())

    st.metric("Average recorded total waste", f"{total:,.1f} kg")

    st.caption(
        "Adjust these hypothetical proportions to explore a scenario."
    )

    wet_pct = st.slider("Wet waste (%)", 0, 100, 50)
    dry_pct = st.slider("Dry recyclable waste (%)", 0, 100, 25)
    sanitary_pct = st.slider("Sanitary / domestic hazardous (%)", 0, 100, 10)
    special_pct = 100 - wet_pct - dry_pct - sanitary_pct

    if special_pct < 0:
        st.error(
            "Your selected percentages exceed 100%. Please reduce "
            "one or more values."
        )
    else:
        categories = pd.DataFrame({
            "Category": [
                "Wet", "Dry recyclable",
                "Sanitary / domestic hazardous", "Special-care"
            ],
            "Illustrative percentage": [
                wet_pct, dry_pct, sanitary_pct, special_pct
            ],
            "Illustrative amount (kg)": [
                total * wet_pct / 100,
                total * dry_pct / 100,
                total * sanitary_pct / 100,
                total * special_pct / 100
            ]
        })

        st.bar_chart(
            categories.set_index("Category")["Illustrative amount (kg)"]
        )
        st.dataframe(categories, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Category Report",
            csv_bytes(categories),
            "illustrative_bin_analysis.csv",
            "text/csv"
        )

# ==========================================================
# COLLECTION PLANNING AND HIGH-WASTE ALERTS
# ==========================================================

elif page == "Collection Planning":
    st.subheader("🚛 Smart Waste Collection Planning")

    threshold = st.number_input(
        "High-waste alert threshold (kg per record)",
        min_value=0.0,
        value=float(df["actual_waste"].mean() * 1.2),
        step=50.0
    )

    summary = (
        df.groupby("area", as_index=False)
        .agg(
            average_waste_kg=("actual_waste", "mean"),
            highest_recorded_kg=("actual_waste", "max"),
            record_count=("actual_waste", "count")
        )
    )

    def get_priority(value):
        if value >= threshold:
            return "HIGH"
        if value >= df["actual_waste"].mean() * 0.8:
            return "MEDIUM"
        return "LOW"

    summary["priority"] = summary["average_waste_kg"].apply(get_priority)

    summary["recommendation"] = summary["priority"].map({
        "HIGH": "Review collection capacity and pickup frequency.",
        "MEDIUM": "Monitor the area and maintain routine collection.",
        "LOW": "Continue routine collection and monitor changes."
    })

    high_areas = summary[summary["priority"] == "HIGH"]
    low_areas = summary[summary["priority"] == "LOW"]

    col1, col2 = st.columns(2)
    col1.metric("High-priority areas", len(high_areas))
    col2.metric("Lower-waste areas", len(low_areas))

    if not high_areas.empty:
        st.error("🚨 High-waste alert: review these recorded areas.")
        st.dataframe(high_areas, use_container_width=True, hide_index=True)
    else:
        st.success("No recorded area exceeds the current alert threshold.")

    st.subheader("Area Waste Comparison")
    st.bar_chart(summary.set_index("area")["average_waste_kg"])

    st.subheader("Collection Priority Table")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.info(
        "Truck assignments and pickup recommendations are planning "
        "suggestions only. There is no live truck GPS connection."
    )

    st.download_button(
        "Download Collection Plan",
        csv_bytes(summary),
        "collection_plan.csv",
        "text/csv"
    )

# ==========================================================
# AREA EXPLORER
# ==========================================================

elif page == "Area Explorer":
    st.subheader("📍 Hyderabad Locality Explorer")

    group = st.selectbox("Location group", list(AREA_GROUPS.keys()))
    area = st.selectbox("Select locality", AREA_GROUPS[group])

    records = area_records(area)

    if records.empty:
        st.warning(
            f"No historical CSV records exist for {area}. "
            "No actual waste value can be displayed for this locality."
        )
    else:
        st.success(f"{len(records)} record(s) found for {area}.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Records", len(records))
        c2.metric(
            "Average recorded waste",
            f"{records['actual_waste'].mean():,.1f} kg"
        )
        c3.metric(
            "Highest recorded waste",
            f"{records['actual_waste'].max():,.1f} kg"
        )

        st.bar_chart(
            records.set_index("date")["actual_waste"]
            if records["date"].notna().any()
            else records["actual_waste"]
        )

        st.dataframe(records, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Area Data",
            csv_bytes(records),
            f"{area.lower().replace(' ', '_')}_data.csv",
            "text/csv"
        )

    st.subheader("Area Data Coverage")
    coverage = pd.DataFrame([
        {
            "area": name,
            "record_count": len(area_records(name)),
            "status": (
                "Has CSV records"
                if not area_records(name).empty
                else "No CSV records"
            )
        }
        for name in ALL_AREAS
    ])

    st.dataframe(coverage, use_container_width=True, hide_index=True)

    st.caption(
        "Locality names are suggestions for app navigation, not "
        "confirmation of official GHMC circle boundaries."
    )

# ==========================================================
# REPORTS AND DOWNLOADS
# ==========================================================

elif page == "Reports & Downloads":
    st.subheader("📥 Download Waste Management Reports")

    selected = st.multiselect(
        "Choose areas",
        dataset_areas,
        default=dataset_areas
    )

    filtered = df[df["area"].isin(selected)].copy()

    st.metric("Records in report", len(filtered))
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Filtered Dataset",
        csv_bytes(filtered),
        "filtered_waste_dataset.csv",
        "text/csv"
    )

    report = (
        filtered.groupby("area", as_index=False)
        .agg(
            average_waste_kg=("actual_waste", "mean"),
            maximum_waste_kg=("actual_waste", "max"),
            total_recorded_kg=("actual_waste", "sum"),
            record_count=("actual_waste", "count")
        )
    )

    st.subheader("Area Summary")
    st.dataframe(report, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Area Summary",
        csv_bytes(report),
        "area_summary.csv",
        "text/csv"
    )

# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":
    st.subheader("🧠 Machine-Learning Model Performance")

    if COMPARISON_FILE.exists():
        try:
            comparison = pd.read_csv(COMPARISON_FILE)
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )
        except Exception as exc:
            st.error(f"Could not read model comparison: {exc}")
    else:
        st.warning(
            "model_comparison.csv was not found. Check whether "
            "train_model.py creates this file."
        )

    if model is not None:
        st.success("Existing trained model is available.")
    else:
        st.error(f"Model unavailable: {model_error}")

    st.markdown("""
    **Metric guide**

    - **MAE:** average absolute prediction error.
    - **RMSE:** gives large prediction errors more weight.
    - **R²:** measures how well the model explains variation
      in the evaluated dataset.
    """)

# ==========================================================
# CUSTOMER CARE
# ==========================================================

elif page == "Customer Care":
    st.subheader("☎️ Customer Care and Feedback")

    st.write(
        "Submit a local feedback entry for demonstration. "
        "This form does not send an email or contact a real help desk."
    )

    with st.form("customer_care_form"):
        customer_name = st.text_input("Name (optional)")
        contact = st.text_input("Contact details (optional)")
        issue_area = st.selectbox("Related area", available_areas)
        issue_type = st.selectbox(
            "Request type",
            [
                "Waste collection delay",
                "Overflowing bin",
                "Waste prediction question",
                "Recycling / bin guidance",
                "Website problem",
                "Other"
            ]
        )
        message = st.text_area("Describe your request")
        send = st.form_submit_button("Prepare Feedback Record")

    if send:
        if not message.strip():
            st.error("Please describe your request.")
        else:
            ticket = pd.DataFrame([{
                "name": customer_name,
                "contact": contact,
                "area": issue_area,
                "request_type": issue_type,
                "message": message,
                "status": "Demo record - not submitted to an authority"
            }])

            st.success("Feedback record prepared. Download it below.")
            st.download_button(
                "Download Feedback Record",
                csv_bytes(ticket),
                "customer_feedback.csv",
                "text/csv"
            )

# ==========================================================
# HELP CENTER
# ==========================================================

elif page == "Help Center":
    st.subheader("🆘 Help Center")

    with st.expander("What is this project?"):
        st.write(
            "An academic project that demonstrates machine-learning "
            "estimation of municipal solid waste generation."
        )

    with st.expander("How do I make a prediction?"):
        st.markdown("""
        1. Open Waste Prediction.
        2. Select a locality.
        3. Enter population, temperature, and previous-day waste.
        4. Click Predict Waste Generation.
        5. Review the result and download the report if needed.
        """)

    with st.expander("Why does a locality have no records?"):
        st.write(
            "The locality is included in the selection list, but "
            "your CSV does not contain historical data for it."
        )

    with st.expander("Are predictions official municipal forecasts?"):
        st.write(
            "No. Predictions are estimates produced by the saved ML "
            "model and depend on the quality and representativeness "
            "of the training data."
        )

    with st.expander("How do I fix a missing package error?"):
        st.write(
            "Check requirements.txt, install the missing package in "
            "your project environment, and restart Streamlit."
        )

    with st.expander("What CSV columns are required?"):
        st.code(
            "date,area,population,temperature,"
            "previous_day_waste,actual_waste"
        )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Intelligent Municipal Solid Waste Prediction | "
    "B.Tech CSE Major Project"
)

st.caption(
    "Generated demo values and illustrative category proportions "
    "are not verified municipal measurements. Map locations, if "
    "shown elsewhere in the project, are approximate unless verified."
)
