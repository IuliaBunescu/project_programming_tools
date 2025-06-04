import os

import joblib
import pandas as pd
import streamlit as st

from src.config import ALGO_TYPES, ML_ALGO_OPTIONS
from src.ml_pipeline import train_model_pipeline
from src.utils import split_columns_by_type


@st.fragment
def column_types():
    if st.button("Show Column Info"):
        df = st.session_state.data
        col_df = df.dtypes.reset_index()
        col_df.columns = ["Column Name", "Data Type"]
        st.dataframe(col_df, height=200)


@st.fragment
def top_n():
    df = st.session_state.data
    st.number_input(
        label="top",
        label_visibility="collapsed",
        help="Select the number of rows to display",
        min_value=1,
        max_value=df.shape[0],
        key="n",
    )
    if st.button("Show Data", key="top_rows"):
        st.dataframe(df.head(st.session_state.n), height=200)


@st.fragment
def bottom_n():
    df = st.session_state.data
    st.number_input(
        label="bot",
        label_visibility="collapsed",
        help="Select the number of rows to display",
        min_value=1,
        max_value=df.shape[0],
        key="k",
    )
    if st.button("Show Data", key="bottom_rows"):
        st.dataframe(df.tail(st.session_state.k), height=200)


@st.fragment
def describe():
    df = st.session_state.data
    if st.button("Show Stats"):
        st.dataframe(df.describe())


@st.fragment
def nulls_removal():
    df = st.session_state.data
    total_nulls_before = df.isnull().sum().sum()
    st.write(f"Total missing values: {total_nulls_before:,}")

    if total_nulls_before == 0:
        st.success("No missing values detected.")
    else:
        if st.button("Remove All Nulls"):
            df_cleaned = df.dropna()
            total_nulls_after = df_cleaned.isnull().sum().sum()

            st.success("Null values removed.")
            st.write(f"**Remaining missing values:** {total_nulls_after:,}")
            st.info(
                f"New shape after removal: {df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} columns"
            )

            st.session_state.data = df_cleaned


@st.fragment
def train_model():
    df = st.session_state.data

    st.multiselect(
        "Select target column",
        options=df.columns.tolist(),
        help="Choose the target column for supervised algorithms .",
        key="target",
        max_selections=1,
    )

    st.multiselect(
        "Select feature columns",
        options=[col for col in df.columns if col not in st.session_state.target],
        help="Choose the columns to use as input features",
        key="feature_cols",
    )

    st.selectbox(
        "Select an algorithm type",
        options=ALGO_TYPES,
        help="Choose the type of algorithm you'd like to use for prediction",
        key="algo",
    )

    if st.button(
        "Train Model",
        disabled=not (st.session_state.algo and st.session_state.feature_cols),
    ):
        algo_type = st.session_state.algo
        model = ML_ALGO_OPTIONS[algo_type]

        num_cols, cat_cols, _ = split_columns_by_type()

        try:
            # Train
            pipeline, metric_msg = train_model_pipeline(
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                model=model,
                task_type=algo_type,
            )

            # Save model pipeline
            joblib.dump(pipeline, "./model/model_pipeline.pkl")

            st.success(f"Training complete! {metric_msg}")
            st.session_state.model_trained = True

        except Exception as e:
            st.error(" An error occurred during training or saving the model.")
            st.exception(e)


@st.fragment()
def predict_target():
    if st.button("Check for Trained Model"):
        st.rerun(scope="fragment")

    if not st.session_state.get("model_trained", False):
        st.warning("Train the model first or click 'Check for Trained Model'.")
        return

    try:
        pipeline = joblib.load("./model/model_pipeline.pkl")

        df = st.session_state.prediction_row.copy()
        feature_cols = st.session_state.feature_cols

        input_df = df[feature_cols].copy()
        num_cols, cat_cols, cat_options = split_columns_by_type()

        st.subheader("Adjust input values")

        user_input = {}
        columns_per_row = 3
        rows = [
            feature_cols[i : i + columns_per_row]
            for i in range(0, len(feature_cols), columns_per_row)
        ]

        for row in rows:
            cols = st.columns(len(row))
            for i, col in enumerate(row):
                val = input_df[col].values[0]
                if col in num_cols:
                    user_input[col] = cols[i].number_input(col, value=float(val))
                elif col in cat_cols:
                    options = cat_options.get(col, [])
                    default_index = options.index(val) if val in options else 0
                    user_input[col] = cols[i].selectbox(
                        col, options, index=default_index
                    )
                else:
                    user_input[col] = cols[i].text_input(col, value=str(val))

        input_data = pd.DataFrame([user_input])

        predict_disabled = input_data.isnull().any().any()

        if st.button("Predict", disabled=predict_disabled):
            prediction = pipeline.predict(input_data)[0]
            st.subheader("Prediction Result")

            if isinstance(prediction, (int, float)):
                st.write(f"Predicted target value: **{round(prediction, 2)}**")
            else:
                st.write(f"Predicted target value: **{prediction}**")

        if predict_disabled:
            st.warning(
                "Some input values are missing. Please fill in all fields to enable prediction."
            )

    except Exception as e:
        st.error("An error occurred during prediction.")
        st.exception(e)
