import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Customer Segmentation App")

# Input section
st.header("Predict Customer Segment")
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

# Visualization section
st.header("Customer Segment Distribution")

try:
    df = pd.read_csv("segmented_customers.csv")

    # Bar chart
    st.subheader("Number of Customers per Segment")
    fig, ax = plt.subplots()
    df['Segment'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Count by Segment")
    st.pyplot(fig)

    # Scatter plot: Income vs Segment
    st.subheader("Income vs Segment")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['Segment'], df['Income'], c=df['Segment'], cmap='tab10', alpha=0.6, edgecolors='k')
    ax2.set_xlabel("Segment")
    ax2.set_ylabel("Income")
    ax2.set_title("Customer Income by Segment")
    st.pyplot(fig2)

except FileNotFoundError:
    st.warning("File 'segmented_customers.csv' not found. Please generate it first.")
