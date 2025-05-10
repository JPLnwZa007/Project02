import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🧠 Customer Segmentation App")

# Input section
st.header("🔍 Predict Customer Segment")
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
    st.success(f"🎯 Predicted Customer Segment: {segment[0]}")

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
    df['Segment Name'] = df['Segment'].map(segment_names)

    # Bar Chart
    st.subheader("🧮 Segment Counts")
    segment_counts = df['Segment Name'].value_counts().sort_index()

    fig, ax = plt.subplots()
    bars = ax.bar(segment_counts.index, segment_counts.values, color=['#4e79a7', '#f28e2c', '#e15759', '#76b7b2'])

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Count by Segment")
    st.pyplot(fig)

    # Scatter Plot
    st.subheader("📈 Income vs Recency by Segment")
    fig2, ax2 = plt.subplots()

    colors = {
        "Budget": "#4e79a7",
        "Premium": "#f28e2c",
        "Young": "#e15759",
        "Loyal": "#76b7b2"
    }

    for segment_name, color in colors.items():
        segment_data = df[df['Segment Name'] == segment_name]
        ax2.scatter(segment_data['Income'], segment_data['Recency'],
                    label=segment_name, color=color, alpha=0.7, edgecolors='k')

    ax2.set_xlabel("Income")
    ax2.set_ylabel("Recency")
    ax2.set_title("Income vs Recency by Customer Segment")
    ax2.legend(title="Segment")
    st.pyplot(fig2)

except FileNotFoundError:
    st.warning("⚠️ File 'segmented_customers.csv' not found. Please run the training script first.")
