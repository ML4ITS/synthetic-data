description = "Example app 1"

# Your app goes in the function run()
def run():
        
    import streamlit as st

    st.header("Example app 1")
    st.write("Insert your own Streamlit code here")

# end of app

# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()