import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Input section
st.header("Predict Customer Segment")
income_range = st.slider("Income Range", 0, 200000, (1000, 2000))
recency = st.number_input("Recency", min_value=0)
wines = st.number_input("Monthly Wine Spend", min_value=0)

if st.button("Predict Segment"):
    # Use the average of the income range
    income = sum(income_range) / 2
    data = [[income, recency, wines]]
    data_scaled = scaler.transform(data)
    segment = model.predict(data_scaled)
    st.success(f"Predicted Customer Segment: {segment[0]}")

# Visualization section
st.header("Customer Segment Distribution")

try:
    df = pd.read_csv("segmented_customers.csv")

    # Check if necessary columns exist
    if 'Income' in df.columns and 'Segment' in df.columns:
        # Plot income distribution by segment
        plt.figure(figsize=(10, 6))
        for segment in sorted(df['Segment'].unique()):
            subset = df[df['Segment'] == segment]
            plt.hist(subset['Income'], bins=30, alpha=0.5, label=f"Segment {segment}")
        plt.xlabel("Income")
        plt.ylabel("Number of Customers")
        plt.title("Income Distribution by Customer Segment")
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("The file does not contain required 'Income' and 'Segment' columns.")

except FileNotFoundError:
    st.warning("File 'segmented_customers.csv' not found. Please generate it first.")

