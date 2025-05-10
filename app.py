import streamlit as st
import pandas as pd
import pickle

# Load model and data
model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Customer Segmentation App")

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

