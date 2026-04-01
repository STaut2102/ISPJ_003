import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Restaurant Success", layout="wide")

# --- ฟังก์ชันโหลด Assets (แก้ไขชื่อไฟล์ให้ตรงกับ GitHub) ---
@st.cache_resource
def load_all():
    with open('model_ml.pkl', 'rb') as f: ml = pickle.load(f)
    with open('model_nn.pkl', 'rb') as f: nn = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: sc = pickle.load(f)
    with open('le_cuisine.pkl', 'rb') as f: le = pickle.load(f)
    
    # แก้จุดนี้: เปลี่ยนจาก data_restaurant.csv เป็น restaurant_data.csv ตาม GitHub
    df = pd.read_csv('restaurant_data.csv') 
    return ml, nn, sc, le, df

try:
    ml_model, nn_model, scaler, le_cuisine, df_raw = load_all()
except Exception as e:
    # แสดง Error จริงออกมาดูถ้ายังโหลดไม่ได้
    st.error(f"❌ พบปัญหาการโหลดไฟล์: {e}")
    st.stop()

# --- Sidebar พร้อม Credit ---
st.sidebar.title("📑 AI Restaurant Project")
page = st.sidebar.radio("เมนูหลัก", ["Info: ML Theory & Data", "Info: NN Theory", "Test: ML Predict", "Test: NN Predict"])

st.sidebar.markdown("---")
st.sidebar.caption("🤖 **AI Collaboration Credit**")
st.sidebar.write("Designed with support from **Gemini AI (Google)**")

# --- หน้า 1: อธิบาย ML + Dashboard ---
if page == "Info: ML Theory & Data":
    st.title("🍔 AI พยากรณ์ความสำเร็จร้านอาหาร")
    st.header("📊 ข้อมูลร้านอาหารในระบบ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**จำนวนร้านแยกตามประเภท**")
        st.bar_chart(df_raw['Cuisine'].value_counts())
    with col2:
        st.write("**ราคาเฉลี่ยต่อจาน (บาท)**")
        # ใช้ข้อมูลจาก Dataset 1 ที่ทำ Cleaning แล้ว
        avg_price = df_raw.groupby('Cuisine')['Avg_Price'].mean()
        st.line_chart(avg_price)
    
    st.divider()
    st.write("### ทฤษฎี Machine Learning (Ensemble)")
    st.write("ใช้การทำ Soft Voting เพื่อคำนวณโอกาสสำเร็จเป็นเปอร์เซ็นต์")

# --- หน้า 2: อธิบาย NN ---
elif page == "Info: NN Theory":
    st.title("🧠 ทฤษฎี Neural Network")
    st.write("### โครงสร้างโมเดล (MLP)")
    st.write("ใช้ Hidden Layers (16, 8) ในการประมวลผลแทนโครงสร้างเดิมที่ใช้ LSTM")

# --- หน้า 3: ทดสอบ ML ---
elif page == "Test: ML Predict":
    st.title("🔮 พยากรณ์ด้วย Machine Learning")
    c = st.selectbox("เลือกประเภทอาหาร", le_cuisine.classes_)
    p = st.number_input("ราคาต่อจาน", value=150)
    l = st.slider("ทำเล (1-10)", 1, 10, 7)
    s = st.number_input("ยอด Check-in", value=500)
    
    if st.button("ทำนายผล (ML)"):
        # ต้องผ่าน Scaler ก่อนทำนายเสมอ
        inp = np.array([[le_cuisine.transform([c])[0], p, l, s]])
        scaled = scaler.transform(inp)
        prob = ml_model.predict_proba(scaled)[0][1] * 100
        st.subheader(f"โอกาสสำเร็จ: {prob:.2f}%")
        st.progress(prob/100)

# --- หน้า 4: ทดสอบ NN ---
elif page == "Test: NN Predict":
    st.title("🤖 พยากรณ์ด้วย Neural Network")
    c = st.selectbox("ประเภทอาหาร ", le_cuisine.classes_)
    p = st.number_input("ราคาต่อจาน ", value=150)
    l = st.slider("คะแนนทำเล ", 1, 10, 7)
    s = st.number_input("เช็คอินโซเชียล ", value=500)
    
    if st.button("ทำนายผล (NN)"):
        inp = np.array([[le_cuisine.transform([c])[0], p, l, s]])
        scaled = scaler.transform(inp)
        prob = nn_model.predict_proba(scaled)[0][1] * 100
        st.subheader(f"โอกาสสำเร็จ (AI): {prob:.2f}%")
        st.progress(prob/100)
