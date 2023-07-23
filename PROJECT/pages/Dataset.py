# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# st.set_option('deprecation.showPyplotGlobalUse', False)

# def main():
#     st.title("Dataset Visualization")

#     # Load your dataset here. Replace 'your_dataset.csv' with your actual dataset file.
#     # You can use any method to load data into a pandas DataFrame (e.g., pd.read_csv, pd.read_excel, etc.).
#     df = pd.read_csv('dataset.csv')

#     st.write("## Sample Data")
#     st.write(df.head())  # Display a sample of the dataset

#     st.write("## Data Summary")
#     st.write(df.describe())  # Display summary statistics of the dataset

#     # Let's create some visualizations for the dataset
#     st.write("## Data Visualization")

#     # Example 1: Histogram
#     st.write("### Histogram")
#     column_for_histogram = st.selectbox("Select a column for the histogram", df.columns)
#     plt.hist(df[column_for_histogram], bins=30, alpha=0.7)
#     st.pyplot()

#     # Example 2: Scatter Plot
#     st.write("### Scatter Plot")
#     column_x = st.selectbox("Select X-axis column", df.columns)
#     column_y = st.selectbox("Select Y-axis column", df.columns)
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(data=df, x=column_x, y=column_y)
#     st.pyplot()

#     # Add more visualizations here as needed, such as line plots, bar plots, etc.

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Heart Disease Predictor Dataset Visualization")

    # Load your heart disease dataset here. Replace 'heart_disease_data.csv' with your actual dataset file.
    # You can use any method to load data into a pandas DataFrame (e.g., pd.read_csv, pd.read_excel, etc.).
    df = pd.read_csv('dataset.csv')

    st.write("## Sample Data")
    st.write(df.head())  # Display a sample of the dataset

    st.write("## Data Summary")
    st.write(df.describe())  # Display summary statistics of the dataset

    # Let's create some visualizations for the heart disease predictor dataset
    st.write("## Data Visualization")

    # Example 1: Bar chart of the target variable (Presence or Absence of Heart Disease)
    st.write("### Target Variable Distribution")
    target_counts = df['target'].value_counts()
    st.bar_chart(target_counts)

    # # Example 2: Histogram of Age
    # st.write("### Histogram of Age")
    # plt.hist(df['age'], bins=20, alpha=0.7)
    # plt.xlabel("Age")
    # plt.ylabel("Frequency")
    # st.pyplot()

    # Example 3: Correlation heatmap
    st.write("### Correlation Heatmap")
    correlation = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()
    
    # Example 1: Histogram
    st.write("### Histogram")
    column_for_histogram = st.selectbox("Select a column for the histogram", df.columns)
    plt.hist(df[column_for_histogram], bins=30, alpha=0.7)
    st.pyplot()

    # Example 2: Scatter Plot
    st.write("### Scatter Plot")
    column_x = st.selectbox("Select X-axis column", df.columns)
    column_y = st.selectbox("Select Y-axis column", df.columns)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=column_x, y=column_y)
    st.pyplot()


    # Add more visualizations here as needed, such as scatter plots, bar plots, etc.

if __name__ == "__main__":
    main()
