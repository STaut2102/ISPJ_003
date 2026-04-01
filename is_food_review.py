import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. ตั้งค่าหน้าเว็บให้ดูเป็นมืออาชีพ ---
st.set_page_config(page_title="Restaurant Success Predictor", layout="wide")

# --- 2. ฟังก์ชันโหลด Assets (โมเดลและข้อมูล) ---
@st.cache_resource
def load_all_assets():
    try:
        # โหลดโมเดลและตัวแปลงค่าต่างๆ
        with open('model_ml.pkl', 'rb') as f: ml = pickle.load(f)
        with open('model_nn.pkl', 'rb') as f: nn = pickle.load(f)
        with open('scaler.pkl', 'rb') as f: sc = pickle.load(f)
        with open('le_cuisine.pkl', 'rb') as f: le = pickle.load(f)
        
        # โหลด Dataset 1 (ตรวจสอบชื่อไฟล์ให้ตรงกับ GitHub)
        df = pd.read_csv('restaurant_data.csv')
        
        return ml, nn, sc, le, df
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดไฟล์ได้: {e}")
        st.info("ตรวจสอบว่าไฟล์ .pkl และ .csv ทั้งหมดอัปโหลดขึ้น GitHub ครบถ้วนและชื่อถูกต้อง")
        st.stop()

# เรียกใช้งานฟังก์ชันโหลด
ml_model, nn_model, scaler, le_cuisine, df_raw = load_all_assets()

# --- 3. ส่วนของ Sidebar (เมนู 4 หน้า) ---
st.sidebar.title("🍱 Menu")
page = st.sidebar.radio("เลือกหน้า", [
    "📊 Info: ML Theory & Data", 
    "🧠 Info: NN Theory", 
    "🔮 Test: ML Predict", 
    "🤖 Test: NN Predict"
])

st.sidebar.markdown("---")
st.sidebar.caption("🤖 **AI Collaboration Credit**")
st.sidebar.write("Project designed with **Gemini AI**")

# --- หน้าที่ 1: Dashboard และทฤษฎี ML ---
if page == "📊 Info: ML Theory & Data":
    st.title("🍔 ข้อมูลและการพยากรณ์ด้วย ML")
    
    # ส่วน Dashboard
    st.subheader("📈 วิเคราะห์ข้อมูลเบื้องต้น")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**จำนวนร้านแยกตามประเภท**")
        st.bar_chart(df_raw['Cuisine'].value_counts())
        
    with col2:
        st.write("**ราคาเฉลี่ยต่อจาน (บาท)**")
        # แก้ไขจุดที่ทำให้เกิด TypeError ในกราฟ
        df_raw['Avg_Price'] = pd.to_numeric(df_raw['Avg_Price'], errors='coerce')
        avg_price_chart = df_raw.groupby('Cuisine')['Avg_Price'].mean()
        st.line_chart(avg_price_chart)

    st.divider()
    st.write("### 📝 ทฤษฎี Machine Learning")
    st.markdown("""
    - **Model Type:** Ensemble Voting Classifier
    - **Algorithms:** Random Forest, XGBoost และ Logistic Regression
    - **Process:** ใช้ระบบ Soft Voting เพื่อคำนวณหาค่าเฉลี่ยความน่าจะเป็น (Probability) ออกมาเป็นเปอร์เซ็นต์
    """)

# --- หน้าที่ 2: ทฤษฎี Neural Network ---
elif page == "🧠 Info: NN Theory":
    st.title("🧠 โครงสร้าง Neural Network")
    st.write("### Multi-layer Perceptron (MLP)")
    st.markdown("""
    - **Architecture:** 2 Hidden Layers (16 และ 8 Neurons)
    - **Activation Function:** ReLU สำหรับชั้นซ่อน และ Sigmoid สำหรับ Output
    - **Optimization:** ใช้ 'Adam' optimizer ในการฝึกฝนโมเดล
    - **Note:** โมเดลนี้ถูกออกแบบมาให้เบาและรันได้รวดเร็วบนระบบ Cloud
    """)

# --- หน้าที่ 3: ทดสอบพยากรณ์ด้วย ML ---
elif page == "🔮 Test: ML Predict":
    st.title("🔮 ทดสอบพยากรณ์ (ML)")
    
    with st.container(border=True):
        c = st.selectbox("ประเภทอาหาร", le_cuisine.classes_)
        p = st.number_input("ราคาเฉลี่ย (บาท)", min_value=0, value=150)
        l = st.slider("ระดับทำเล (1-10)", 1, 10, 5)
        s = st.number_input("ยอด Check-in โซเชียล", min_value=0, value=100)
        
        if st.button("ประมวลผลด้วย ML", use_container_width=True):
            # เตรียมข้อมูลและทำ Scaling
            inp = np.array([[le_cuisine.transform([c])[0], p, l, s]])
            inp_scaled = scaler.transform(inp)
            
            # ทำนายเป็น %
            prob = ml_model.predict_proba(inp_scaled)[0][1] * 100
            
            st.subheader(f"โอกาสรอดของร้านคือ: {prob:.2f}%")
            st.progress(prob/100)

# --- หน้าที่ 4: ทดสอบพยากรณ์ด้วย NN ---
elif page == "🤖 Test: NN Predict":
    st.title("🤖 ทดสอบพยากรณ์ (NN AI)")
    
    with st.container(border=True):
        c = st.selectbox("ประเภทอาหาร ", le_cuisine.classes_)
        p = st.number_input("ราคาเฉลี่ยต่อจาน ", min_value=0, value=150)
        l = st.slider("ความสวยงามของทำเล (1-10) ", 1, 10, 5)
        s = st.number_input("ยอดผู้เข้าใช้งาน (Social) ", min_value=0, value=100)
        
        if st.button("ประมวลผลด้วย Neural Network", use_container_width=True):
            # เตรียมข้อมูลและทำ Scaling
            inp = np.array([[le_cuisine.transform([c])[0], p, l, s]])
            inp_scaled = scaler.transform(inp)
            
            # ทำนายเป็น % ด้วย MLP
            prob = nn_model.predict_proba(inp_scaled)[0][1] * 100
            
            st.subheader(f"AI วิเคราะห์โอกาสรอดได้: {prob:.2f}%")
            st.progress(prob/100)
