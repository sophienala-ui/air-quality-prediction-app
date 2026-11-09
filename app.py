import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
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

st.title("🌍 Air Quality Prediction")
st.subheader("SDG 11: Sustainable Cities and Communities")
st.write(
    "Estimate PM2.5 concentration using pollution measurements, "
    "weather conditions, time information, and monitoring-station location."
)

st.info(
    "This app estimates PM2.5 for an observation using measurements recorded "
    "at the same time. It is not a forecast of future air quality and does not "
    "calculate an official AQI."
)

@st.cache_data
def load_data():
    air_quality = fetch_ucirepo(id=501)

    if air_quality.data.original is not None:
        df = air_quality.data.original.copy()
    else:
        df = pd.concat(
            [air_quality.data.features, air_quality.data.targets],
            axis=1
        )

    df.columns = [str(c).strip() for c in df.columns]
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
    categorical_features = [c for c in categorical_features if c in df.columns]
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

with st.spinner("Loading the UCI air-quality dataset..."):
    df = load_data()

with st.spinner("Training the Random Forest model..."):
    result = train_model(df)

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
                        feature, min_value=1, max_value=12,
                        value=max(1, min(12, default_value)), step=1
                    )
                elif feature == "day":
                    input_data[feature] = st.number_input(
                        feature, min_value=1, max_value=31,
                        value=max(1, min(31, default_value)), step=1
                    )
                elif feature == "hour":
                    input_data[feature] = st.number_input(
                        feature, min_value=0, max_value=23,
                        value=max(0, min(23, default_value)), step=1
                    )
                else:
                    input_data[feature] = st.number_input(
                        feature,
                        value=default_value,
                        step=1
                    )

        for feature in categorical_features:
            options = sorted(df[feature].dropna().astype(str).unique().tolist())
            input_data[feature] = st.selectbox(feature, options)

    if st.button("Predict PM2.5", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.success("Prediction completed.")
        st.metric("Predicted PM2.5", f"{prediction:.2f}")

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
    st.header("Dataset")

    st.write(
        "Source: Beijing Multi-Site Air-Quality Data from the "
        "UCI Machine Learning Repository."
    )

    c1, c2 = st.columns(2)
    c1.metric("Rows after target/timestamp cleaning", f"{len(df):,}")
    c2.metric("Number of columns", f"{len(df.columns):,}")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Average PM2.5 by Month")

    monthly_avg = (
        df.groupby("month")["PM2.5"]
        .mean()
        .reindex(range(1, 13))
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    monthly_avg.plot(marker="o", ax=ax)
    ax.set_title("Average PM2.5 by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average PM2.5 concentration")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    st.pyplot(fig)

st.divider()
st.caption(
    "Project: Air Quality Prediction Using Machine Learning | SDG 11"
)
