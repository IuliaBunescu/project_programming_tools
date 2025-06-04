import pandas as pd
import streamlit as st


# Function to load CSS from the 'assets' folder
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def split_columns_by_type():
    columns = st.session_state.feature_cols
    df = st.session_state.data

    numerical_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in columns if col not in numerical_cols]

    categorical_options = {
        col: df[col].dropna().unique().tolist() for col in categorical_cols
    }

    return numerical_cols, categorical_cols, categorical_options


def add_vertical_space(num_lines: int = 1) -> None:
    for _ in range(num_lines):
        st.write("")
