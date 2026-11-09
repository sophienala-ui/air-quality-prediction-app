import io
import zipfile
import urllib.request

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="Air Quality Prediction | SDG 11",
    page_icon="🌍",
    layout="wide"
)


# Custom HTML/CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        margin-top: 0;
        opacity: 0.8;
    }

    .info-card {
        padding: 18px 22px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.08);
        margin: 12px 0 22px 0;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
    }

    .small-note {
        font-size: 14px;
        opacity: 0.75;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">🌍 Air Quality Prediction</div>
<div class="subtitle">SDG 11 — Sustainable Cities and Communities</div>

<div class="info-card">
Estimate PM2.5 concentration using pollution measurements, weather conditions,
time information, and monitoring-station location.
</div>
""", unsafe_allow_html=True)

st.info(
    "This app estimates PM2.5 for an observation using measurements recorded "
    "at the same time. It is not a forecast of future air quality and does not "
    "calculate an official AQI."
)

# UCI's official downloadable archive for dataset 501.
# We download the archive directly instead of using ucimlrepo.fetch_ucirepo(),
# because that importer is currently reporting dataset 501 as unavailable.
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing+multi+site+air+quality.zip"
)


@st.cache_data(show_spinner=False)
def load_data():
    request = urllib.request.Request(
        UCI_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
            and "PRSA_Data_" in name
        ]

        if not csv_files:
            raise RuntimeError(
                "The UCI archive was downloaded, but the expected PRSA CSV "
                "files were not found."
            )

        frames = []
        for name in csv_files:
            with z.open(name) as f:
                station_df = pd.read_csv(f)
                frames.append(station_df)

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]

    # The source files already contain the station column, but this keeps
    # the app robust if a source file has a different filename convention.
    if "station" not in df.columns:
        station_names = []
        for name in csv_files:
            station = name.replace("PRSA_Data_", "").split("_")[0]
            station_names.append(station)

        if len(station_names) == len(frames):
            for station_name, frame in zip(station_names, frames):
                frame["station"] = station_name
            df = pd.concat(frames, ignore_index=True)

    return df


@st.cache_resource
def train_model(df):
    df = df.copy()

    numeric_candidates = [
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        "year", "month", "day", "hour"
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]],
        errors="coerce"
    )

    df = df.dropna(subset=["PM2.5", "timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    target = "PM2.5"

    numeric_features = [
        "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        "year", "month", "day", "hour"
    ]

    categorical_features = ["wd", "station"]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [
        c for c in categorical_features if c in df.columns
    ]
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns].copy()
    y = df[target].copy()

    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "df": df,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


try:
    with st.spinner("Downloading the UCI air-quality dataset..."):
        df = load_data()

    with st.spinner("Training the Random Forest model..."):
        result = train_model(df)

except Exception as e:
    st.error("The app could not load the UCI dataset.")
    st.exception(e)
    st.stop()


model = result["model"]
feature_columns = result["feature_columns"]
numeric_features = result["numeric_features"]
categorical_features = result["categorical_features"]
df = result["df"]

st.success("Model is ready.")

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Performance", "Dataset"])


with tab1:
    st.header("Predict PM2.5")
    st.write("Enter the measurements for one observation.")

    input_data = {}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pollution & Weather")

        for feature in numeric_features:
            if feature in ["year", "month", "day", "hour"]:
                continue

            default_value = float(df[feature].median())

            input_data[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.3f"
            )

    with col2:
        st.subheader("Time & Location")

        for feature in ["year", "month", "day", "hour"]:
            if feature in feature_columns:
                default_value = int(df[feature].median())

                if feature == "month":
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=1,
                        max_value=12,
                        value=max(1, min(12, default_value)),
                        step=1
                    )
                elif feature == "day":
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=1,
                        max_value=31,
                        value=max(1, min(31, default_value)),
                        step=1
                    )
                elif feature == "hour":
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=0,
                        max_value=23,
                        value=max(0, min(23, default_value)),
                        step=1
                    )
                else:
                    input_data[feature] = st.number_input(
                        feature,
                        value=default_value,
                        step=1
                    )

        for feature in categorical_features:
            options = sorted(
                df[feature].dropna().astype(str).unique().tolist()
            )
            input_data[feature] = st.selectbox(feature, options)

    if st.button(
        "Predict PM2.5",
        type="primary",
        use_container_width=True
    ):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.success("Prediction completed.")
        st.metric("Predicted PM2.5", f"{prediction:.2f} µg/m³")


with tab2:
    st.header("Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{result['mae']:.3f}")
    c2.metric("RMSE", f"{result['rmse']:.3f}")
    c3.metric("R²", f"{result['r2']:.3f}")

    st.write(
        "The model uses the first 80% of the chronological observations for "
        "training and the final 20% for testing, following the notebook."
    )

    y_test = result["y_test"]
    predictions = result["predictions"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, predictions, alpha=0.25)
    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title("Actual vs Predicted PM2.5")
    ax.set_xlabel("Actual PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    fig.tight_layout()
    st.pyplot(fig)


with tab3:
    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)

    st.write(
        "Source: Beijing Multi-Site Air-Quality Data from the "
        "UCI Machine Learning Repository."
    )

    c1, c2 = st.columns(2)
    c1.metric(
        "Rows after target/timestamp cleaning",
        f"{len(df):,}"
    )
    c2.metric("Number of columns", f"{len(df.columns):,}")

    st.dataframe(df.head(20), use_container_width=True)
