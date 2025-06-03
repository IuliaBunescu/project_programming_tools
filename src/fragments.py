import streamlit as st


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
        label="",
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
        label="",
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
def get_prediction_input():
    df = st.session_state.data

    feature_cols = st.multiselect(
        "Select feature columns",
        options=df.columns.tolist(),
        help="Choose the columns to use as input features",
        key="feature_cols",
    )

    algorithm = st.selectbox(
        "Select an algorithm type",
        options=["Regression", "Classification", "Clustering"],
        help="Choose the type of algorithm you'd like to use for prediction",
        key="algo",
    )
