import streamlit as st
import pickle
import numpy as np
import base64


st.set_page_config(page_title="GlucoSense AI 💖", layout="centered", page_icon="💗")


def add_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Poppins', sans-serif;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(14px);
        }}

        [data-testid="stSidebar"] h1 {{
            font-size: 1.8rem;
            color: #ff69b4;
            text-align: center;
            font-weight: 800;
            text-shadow: 0 0 20px rgba(255,105,180,0.6);
        }}

        /* Glass Card */
        .glass {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 25px;
            padding: 2.5rem;
            backdrop-filter: blur(15px);
            box-shadow: 0 4px 40px rgba(255, 182, 193, 0.35);
            animation: fadeIn 1s ease-in-out;
        }}

        @keyframes fadeIn {{
            from {{opacity: 0; transform: translateY(10px);}}
            to {{opacity: 1; transform: translateY(0);}}
        }}

        h1 {{
            text-align: center;
            font-size: 2.5rem;
            color: #ffffff;
            text-shadow: 0 0 15px #ffb6c1;
        }}

        h3, label {{
            color: #fff;
            font-size: 1.2rem;
        }}

        p {{
            color: #fff;
            font-size: 1rem;
        }}

        .stButton>button {{
            width: 100%;
            background: linear-gradient(135deg,#ff85b3,#ff5fa2);
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            border: none;
            transition: 0.3s;
        }}

        .stButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 0 25px #ff8ab8;
        }}

        .result-success {{
            text-align:center;
            font-size:1.6rem;
            color:#b9ffd9;
            text-shadow:0 0 15px #7fffca;
            font-weight:700;
            animation: fadeResult 1.2s ease-in-out;
        }}

        .result-error {{
            text-align:center;
            font-size:1.6rem;
            color:#ff9f9f;
            text-shadow:0 0 15px #ff7a7a;
            font-weight:700;
            animation: fadeResult 1.2s ease-in-out;
        }}

        @keyframes fadeResult {{
            from {{opacity: 0; transform: scale(0.9);}}
            to {{opacity: 1; transform: scale(1);}}
        }}

        .title-logo {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
        }}
        .title-logo img {{
            width: 48px;
            height: 48px;
            filter: drop-shadow(0 0 8px #ffb6c1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("bg.jpg")


model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# navbar
st.sidebar.title("💖 GlucoSense AI")
page = st.sidebar.radio("Navigate", ["🏠 Home", "💡 About", "⚙️ How It Works"])

# home
if page == "🏠 Home":
    st.markdown("""
    <div class="title-logo">
        <img src="https://cdn-icons-png.flaticon.com/512/2947/2947560.png">
        <h1>GlucoSense AI</h1>
    </div>
    <p style='text-align:center;font-size:1.2rem;color:#fff;'>Smart Diabetes Prediction System</p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=250, value=100)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("🔍 Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        std_data = scaler.transform(input_data)
        prediction = model.predict(std_data)

        if prediction[0] == 1:
            st.markdown("<p class='result-error'>⚠️ The person is Diabetic</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='result-success'>✅ The person is Non-Diabetic</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#fff;'>Made with 💖 by <b>Srija Chinthakunta</b></p>", unsafe_allow_html=True)

# about
elif page == "💡 About":
    st.markdown("<h1>💡 About GlucoSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("""
    **GlucoSense AI** is an intelligent, user-friendly health prediction tool built to assist  
    users in understanding their risk of diabetes using basic health indicators.  

    🩷 Powered by Machine Learning  
    ⚙️ Built using Python, Scikit-Learn, and Streamlit  
    ✨ Designed for early awareness and health consciousness  

    **Made with care by Srija Chinthakunta**
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# how 
elif page == "⚙️ How It Works":
    st.markdown("<h1>⚙️ How It Works</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("""
    1. **Input:** The user provides 8 medical parameters.  
    2. **Standardization:** Data is normalized using the same scaler used during training.  
    3. **Prediction:** The trained ML model processes input and predicts the outcome.  
    4. **Result:** A beautiful, color-coded response appears instantly.  

    🧠 GlucoSense AI empowers users to take proactive health steps —  
    with simplicity, speed, and accuracy.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
