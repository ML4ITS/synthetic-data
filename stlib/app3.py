description = "Time Series Visualization"

# Your app goes in the function run()
def run():

    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")
    import seaborn as sns

    st.subheader("Exploratory Data Analysis")

    data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())

        if st.checkbox("Show Shape"):
            st.write(df.shape)

        if st.checkbox("Show Columns"):
            all_columns = df.columns.to_list()
            st.write(all_columns)

        if st.checkbox("Summary"):
            st.write(df.describe())

        if st.checkbox("Show Selected Columns"):
            selected_columns = st.multiselect("Select Columns", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("Show Value Counts"):
            st.write(df.iloc[:, -1].value_counts())

        if st.checkbox("Correlation Plot(Matplotlib)"):
            plt.matshow(df.corr())
            st.pyplot()

        if st.checkbox("Correlation Plot(Seaborn)"):
            st.write(sns.heatmap(df.corr(), annot=True))
            st.pyplot()

        if st.checkbox("Pie Plot"):
            all_columns = df.columns.to_list()
            column_to_plot = st.selectbox("Select 1 Column", all_columns)
            pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot()


# end of app

# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()
