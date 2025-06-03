import streamlit as st

st.set_page_config(
    page_title="Julia's Project",
    page_icon="✨",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("Fast ML")
    if not st.user.is_logged_in:
        st.write("But first, let's log in ...")
        st.button("Authenticate with Google", on_click=st.login, args=["google"])

    else:
        st.title(f"Welcome, {st.user.name}!")
        st.write(f"Email: {st.user.email}")
        st.image(st.user.picture)
        if st.button("Log out"):
            st.logout()
