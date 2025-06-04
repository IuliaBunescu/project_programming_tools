import pathlib

import streamlit as st

import src.fragments as frag
import src.sections as sec
from src.utils import add_vertical_space, load_css

st.set_page_config(
    page_title="Julia's Project",
    page_icon="✨",
    initial_sidebar_state="expanded",
    layout="wide",
)
CURRENT_DIR = pathlib.Path(__file__).parent
css_path = CURRENT_DIR / "assets" / "style.css"
load_css(css_path)

if "initialized" not in st.session_state:
    st.session_state.valid_data = False
    st.session_state.new_data = False
    st.session_state.initialized = True


def reset_data():
    st.session_state.algo = ""
    st.session_state.feature_cols = []
    st.session_state.target = []
    st.session_state.valid_data = False
    st.session_state.model_trained = False


def main():
    with st.sidebar:
        st.title("Fast ML")
        st.divider()
        if not st.user.is_logged_in:
            sec.user_login()
        else:
            sec.upload_file()
            if st.session_state.valid_data:
                st.button(
                    "Process Data",
                    key="new_data",
                    use_container_width=True,
                    type="primary",
                    on_click=reset_data(),
                )

            st.divider()
            add_vertical_space(12)
            sec.user_info()

    if not st.user.is_logged_in:
        st.warning("Please log in first.")
        st.stop()

    if st.session_state.new_data:
        eda_tab, ml_tab, pred_tab = st.tabs(["EDA", "Train", "Predict"])
        with eda_tab:
            st.title("Data Exploration")
            sec.data_shape()
            with st.expander("Column Data"):
                frag.column_types()
            with st.expander("Top Rows"):
                frag.top_n()
            with st.expander("Bottom Rows"):
                frag.bottom_n()
            with st.expander("Data Stats"):
                frag.describe()
            with st.expander("Null Data"):
                frag.nulls_removal()
        with ml_tab:
            st.title("Model Training")
            frag.train_model()
        with pred_tab:
            st.title("Make a Prediction")
            frag.predict_target()


if __name__ == "__main__":
    main()
