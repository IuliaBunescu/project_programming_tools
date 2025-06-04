import pandas as pd
import streamlit as st


def user_login():
    st.write("But first, let's log in ...")
    st.button(
        "Authenticate with Google",
        on_click=st.login,
        args=["google"],
        use_container_width=True,
        type="primary",
    )


def user_info():
    with st.container(key="user-info"):
        st.write(f"Welcome, *{st.user.name}*!")
        st.write(f"Email: {st.user.email}")
        st.image(st.user.picture)
        if st.button("Log out", use_container_width=True, type="secondary"):
            st.logout()


def upload_file():
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload the data you want to use for further processing.",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.session_state.data = df
            st.session_state.valid_data = True

        except Exception as e:
            st.error(f"Error reading file. Try a different file.")
    else:
        st.info("Awaiting CSV file upload...")


def data_shape():
    df = st.session_state.data
    rows, cols = df.shape

    col1, col2 = st.columns(2)
    col1.metric(label="Rows", value=f"{rows:,}", border=True)
    col2.metric(label="Columns", value=f"{cols:,}", border=True)
