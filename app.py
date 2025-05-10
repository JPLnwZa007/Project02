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

    # แปลงชื่อ segment ให้อ่านง่าย
    segment_names = {
        1: "Premium",
        2: "Young",
        3: "Loyal"
    }
    df['Segment Name'] = df['Segment'].map(segment_names)

    # สร้างกราฟ
    segment_counts = df['Segment Name'].value_counts().sort_index()

    fig, ax = plt.subplots()
    bars = ax.bar(segment_counts.index, segment_counts.values, color=[ '#f28e2c', '#e15759', '#76b7b2'])

    # ใส่จำนวนบนแท่ง
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(1, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Count by Segment")
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("⚠️ File 'segmented_customers.csv' not found. Please run the training script first.")


