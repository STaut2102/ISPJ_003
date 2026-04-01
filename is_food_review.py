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
    "📊 เนื้อหาข้อมูล Machine Learning", 
    "🧠 เนื้อหาข้อมูล Neural Network", 
    "🔮 ทดลองแบบ Machine Learning", 
    "🤖 ทดลองแบบ Neural Network"
])

st.sidebar.markdown("---")
st.sidebar.caption("🤖 **AI Collaboration Credit**")
st.sidebar.write("Project designed with **Gemini AI**")

# --- หน้าที่ 1: Dashboard และทฤษฎี ML ---
if page == "📊 เนื้อหาข้อมูล Machine Learning":
    st.title("🍔 ข้อมูลและการพยากรณ์ด้วย Machine Learning")
    
    # ส่วน Dashboard
    st.subheader("📈 วิเคราะห์ข้อมูลเบื้องต้น")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**จำนวนร้านแยกตามประเภท**")
        st.bar_chart(df_raw['Cuisine'].value_counts())
        
    with col2:
        st.write("**ราคาเฉลี่ยต่อจาน (บาท)**")
        df_raw['Avg_Price'] = pd.to_numeric(df_raw['Avg_Price'], errors='coerce')
        avg_price_chart = df_raw.groupby('Cuisine')['Avg_Price'].mean()
        st.line_chart(avg_price_chart)

    st.divider()
    
    # เพิ่มเนื้อหาแนวทางการพัฒนา ML
    st.header("📝 แนวทางการพัฒนา (Machine Learning)")
    
    st.subheader("1. การเตรียมข้อมูล (Data Preparation)")
    st.markdown("""
    - **Cleaning:** จัดการค่าว่างด้วย Median/Mean และลบสัญลักษณ์พิเศษในราคาเพื่อให้เป็นตัวเลขที่คำนวณได้
    - **Encoding:** ใช้ *Label Encoding* เพื่อเปลี่ยนชื่อประเภทอาหารให้เป็นรหัสตัวเลข
    - **Scaling:** ใช้ *StandardScaler* ปรับสเกลข้อมูลทุกตัวแปรให้สมดุล เพื่อไม่ให้ยอด Check-in ที่มีค่าสูงบดบังปัจจัยอื่น
    """)
    
    st.subheader("2. ทฤษฎีอัลกอริทึม (Algorithm Theory)")
    st.markdown("""
    - **Model Type:** Ensemble Voting Classifier
    - **Algorithms:** การรวมพลังของ 3 โมเดลหลักคือ *Random Forest*, *XGBoost* และ *Logistic Regression*
    - **Process:** ใช้ระบบ **Soft Voting** เพื่อหาค่าเฉลี่ยความน่าจะเป็นจากทั้ง 3 โมเดล ทำให้ผลลัพธ์มีความแม่นยำสูงและออกมาเป็นเปอร์เซ็นต์ (%)
    """)
    st.info("📚 แหล่งอ้างอิง: Scikit-learn & XGBoost Documentation")

# --- หน้าที่ 2: ทฤษฎี Neural Network ---
elif page == "🧠 เนื้อหาข้อมูล Neural Network":
    st.title("🧠 โครงสร้างและแนวทาง Neural Network")
    
    st.header("📝 แนวทางการพัฒนา (Neural Network)")
    
    st.subheader("1. แนวทางการพัฒนา")
    st.write("เลือกใช้โครงสร้าง *Multi-layer Perceptron (MLP)* แทนที่ LSTM เพื่อให้โมเดลมีขนาดเบา ประมวลผลได้รวดเร็ว และเสถียรเมื่อทำงานบนระบบ Cloud")
    
    st.subheader("2. ทฤษฎีและโครงสร้างโมเดล (Architecture)")
    st.markdown("""
    - **Architecture:** ประกอบด้วย 2 ชั้นซ่อน (Hidden Layers) ขนาด 16 และ 8 Neurons
    - **Activation Function:** ใช้ *ReLU* ในชั้นซ่อนเพื่อเรียนรู้ความสัมพันธ์ที่ซับซ้อน และ *Sigmoid* ในส่วนสุดท้ายเพื่อคำนวณโอกาสสำเร็จเป็นเปอร์เซ็นต์
    - **Optimization:** ใช้ *Adam Optimizer* ในการเพิ่มประสิทธิภาพการเรียนรู้ของโมเดล
    """)
    st.info("📚 แหล่งอ้างอิง: Multi-layer Perceptron by Scikit-learn")

# --- หน้าที่ 3: ทดสอบพยากรณ์ด้วย ML ---
elif page == "🔮 ทดลองแบบ Machine Learning":
    st.title("🔮 ทดสอบพยากรณ์ด้วย Machine Learning")
    
    with st.container(border=True):
        c = st.selectbox("ประเภทอาหาร", le_cuisine.classes_)
        p = st.number_input("ราคาเฉลี่ย (บาท)", min_value=0, value=150)
        l = st.slider("ระดับทำเล (1-10)", 1, 10, 5)
        s = st.number_input("ยอด Check-in โซเชียล", min_value=0, value=100)
        
        if st.button("เริ่มประมวลผล (ML)", use_container_width=True):
            inp = np.array([[le_cuisine.transform([c])[0], p, l, s]])
            inp_scaled = scaler.transform(inp)
            prob = ml_model.predict_proba(inp_scaled)[0][1] * 100
            
            st.subheader(f"โอกาสรอดของร้านคือ: {prob:.2f}%")
            st.progress(prob/100)

# --- หน้าที่ 4: ทดสอบพยากรณ์ด้วย NN ---
elif page == "🤖 ทดลองแบบ Neural Network":
    st.title("🤖 ทดสอบพยากรณ์ด้วย Neural Network (AI)")
    
    with st.container(border=True):
        c = st.selectbox("ประเภทอาหาร ", le_cuisine.classes_)
        p = st.number_input("ราคาเฉลี่ยต่อจาน ", min_value=0, value=150)
        l = st.slider("ความสวยงามของทำเล (1-10) ", 1, 10, 5)
        s = st.number_input("ยอดผู้เข้าใช้งาน (Social) ", min_value=0, value=100)
        
        if st.button("เริ่มประมวลผล (NN AI)", use_container_width=True):
            inp = np.array([[le_cuisine.transform([c])[0], p, l, s]])
            inp_scaled = scaler.transform(inp)
            prob = nn_model.predict_proba(inp_scaled)[0][1] * 100
            
            st.subheader(f"AI วิเคราะห์โอกาสรอดได้: {prob:.2f}%")
            st.progress(prob/100)
