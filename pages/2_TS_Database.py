import streamlit as st

# Your app goes in the function run()
def run() -> None:
    container = st.container()
    st.subheader("Synthetic Time-Series Database")

    with st.sidebar:
        st.sidebar.header("Menu")


if __name__ == "__main__":
    run()
