import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🎯 Customer Segmentation App")

# Input section
st.header("🔍 ทำนายกลุ่มลูกค้า")
income = st.number_input("รายได้ (Income)", min_value=0)
kids = st.slider("จำนวนเด็ก (Kids)", 0, 3)
teens = st.slider("จำนวนวัยรุ่น (Teens)", 0, 3)
recency = st.number_input("ระยะเวลาตั้งแต่การซื้อครั้งล่าสุด (Recency)", min_value=0)
wines = st.number_input("ค่าใช้จ่ายไวน์ต่อเดือน (Wine Spend)")
fruits = st.number_input("ค่าใช้จ่ายผลไม้ต่อเดือน (Fruit Spend)")

if st.button("📌 ทำนายกลุ่ม"):
    data = [[income, kids, teens, recency, wines, fruits]]
    data_scaled = scaler.transform(data)
    segment = model.predict(data_scaled)
    st.success(f"กลุ่มลูกค้าที่คาดว่าเป็น: {segment[0]}")

# Visualization section
st.header("📊 การแสดงผลข้อมูลกลุ่มลูกค้า")

try:
    df = pd.read_csv("segmented_customers.csv")

    # Bar Chart
    st.subheader("🧮 จำนวนลูกค้าแต่ละกลุ่ม")
    fig, ax = plt.subplots()
    df['Segment'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel("กลุ่มลูกค้า (Segment)")
    ax.set_ylabel("จำนวนลูกค้า")
    ax.set_title("จำนวนลูกค้าในแต่ละกลุ่ม")
    st.pyplot(fig)

    # Scatter Plot
    st.subheader("📈 รายได้ vs ความถี่การซื้อ")
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(df['Income'], df['Recency'], c=df['Segment'], cmap='tab10', alpha=0.7, edgecolors='k')
    ax2.set_xlabel("รายได้ (Income)")
    ax2.set_ylabel("Recency (วันนับจากการซื้อครั้งล่าสุด)")
    ax2.set_title("กลุ่มลูกค้าตามรายได้และความถี่การซื้อ")
    st.pyplot(fig2)

except FileNotFoundError:
    st.warning("⚠️ ไม่พบไฟล์ 'segmented_customers.csv' กรุณารันสคริปต์ที่สร้างไฟล์นี้ก่อน")
