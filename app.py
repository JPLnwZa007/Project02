import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


# Visualization section
st.header("📊 Customer Segment Distribution")

try:
    df = pd.read_csv("segmented_customers.csv")
    
    # Map segment numbers to names
    segment_names = {
        0: "Budget",
        1: "Premium",
        2: "Young",
        3: "Loyal"
    }

    fig, ax = plt.subplots()
    df['Segment'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Count by Segment")
    st.pyplot(fig)
    
    st.title("Customer Segmentation App")

    # Input section
    st.header("🔍 Predict Segment")
    income = st.number_input("Income", min_value=0)
    kids = st.slider("Number of Kids", 0, 3)
    teens = st.slider("Number of Teens", 0, 3)
    recency = st.number_input("Recency", min_value=0)
    wines = st.number_input("Monthly Wine Spend")
    fruits = st.number_input("Monthly Fruit Spend")

    if st.button("Predict Segment"):
        data = [[income, kids, teens, recency, wines, fruits]]
        data_scaled = scaler.transform(data)
        segment = model.predict(data_scaled)
        st.success(f"Predicted Customer Segment: {segment[0]}")

except FileNotFoundError:
    st.warning("segmented_customers.csv not found. Please run the training script first.")
