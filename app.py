import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Segment name mapping
segment_names = {
    0: "Normal",
    1: "Premium",
    2: "Young",
    3: "Loyal"
}

# Visualization section
st.header("📊 Customer Segment Distribution")

try:
    df = pd.read_csv("segmented_customers.csv")

    # Map segment numbers to names for better readability
    df['Segment_Name'] = df['Segment'].map(segment_names)

    # Bar chart of customer counts by segment name
    fig, ax = plt.subplots()
    df['Segment_Name'].value_counts().reindex(segment_names.values()).plot(kind='bar', ax=ax)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Count by Segment")
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("segmented_customers.csv not found. Please run the training script first.")

# App title
st.title("🧠 Customer Segmentation App")

# Input section
st.header("🔍 Predict Segment")
with st.form("predict_form"):
    income = st.number_input("Income", min_value=0, help="Monthly income in dollars")
    kids = st.slider("Number of Kids", 0, 3)
    teens = st.slider("Number of Teens", 0, 3)
    recency = st.number_input("Recency", min_value=0, help="Days since last purchase")
    wines = st.number_input("Monthly Wine Spend", min_value=0)
    fruits = st.number_input("Monthly Fruit Spend", min_value=0)

    submitted = st.form_submit_button("Predict Segment")

    if submitted:
        try:
            # Check for empty inputs (optional)
            if income == 0 or recency == 0:
                st.warning("Please fill in all required fields before predicting.")
            else:
                data = [[income, kids, teens, recency, wines, fruits]]
                data_scaled = scaler.transform(data)
                segment = model.predict(data_scaled)
                segment_label = segment_names.get(segment[0], "Unknown")
                st.success(f"Predicted Customer Segment: {segment_label} ({segment[0]})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
