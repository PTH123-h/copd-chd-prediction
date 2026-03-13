import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration (设置为宽屏模式 wide)
st.set_page_config(page_title="COPD-CHD Risk Predictor", page_icon="🩺", layout="wide")

# 2. Header Section
st.title("🩺 CHD Risk Prediction Model for Elderly COPD Patients")
st.markdown("""
**Description:** This interactive calculator predicts the risk of concurrent Coronary Heart Disease (CHD) in elderly patients with Chronic Obstructive Pulmonary Disease (COPD) using a Random Forest algorithm.
***Disclaimer:** This tool is strictly for academic research and validation purposes. It should not be utilized as a substitute for professional clinical judgment.*
""")
st.divider()

# 3. Load the Model
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# 4. Create Two Main Columns: Left (Inputs) and Right (Results)
col_input, col_result = st.columns([1.2, 1], gap="large")

# --- LEFT COLUMN: INPUT FORM ---
with col_input:
    st.subheader("Patient Clinical Characteristics")
    with st.form("prediction_form"):
        # 将左侧的8个变量进一步分为两列，节省高度
        c1, c2 = st.columns(2)
        
        with c1:
            gender_option = st.selectbox('Gender', ['Male', 'Female'])
            gender = 1 if gender_option == 'Male' else 0
            ua = st.number_input('Uric Acid (μmol/L)', min_value=0.0, max_value=1000.0, value=350.0)
            hs_crp = st.number_input('hs-CRP (mg/L)', min_value=0.0, max_value=200.0, value=5.0)
            hs_ctni = st.number_input('hs-cTnI (pg/mL)', min_value=0.0, max_value=10000.0, value=15.0)
            
        with c2:
            age = st.number_input('Age (years)', min_value=40, max_value=120, value=75)
            glu = st.number_input('Fasting Glucose (mmol/L)', min_value=0.0, max_value=50.0, value=5.5)
            bun = st.number_input('Blood Urea Nitrogen (mmol/L)', min_value=0.0, max_value=100.0, value=6.0)
            tt = st.number_input('Thrombin Time (s)', min_value=0.0, max_value=100.0, value=18.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="Calculate CHD Risk", use_container_width=True)

# --- RIGHT COLUMN: RESULTS ---
with col_result:
    st.subheader("Prediction Result")
    
    # 只有当用户点击了左侧的计算按钮时，右侧才会显示结果
    if submit_button:
        # Prepare data
        data = {
            'GENDER': gender,
            'UA': ua,
            'hs_CRP': hs_crp,
            'Glu': glu,
            'AGE': age,
            'BUN': bun,
            'hs_cTnI': hs_ctni,
            'TT': tt
        }
        features = pd.DataFrame(data, index=[0])
        cols_order = ['GENDER', 'UA', 'hs_CRP', 'Glu', 'AGE', 'BUN', 'hs_cTnI', 'TT']
        features = features[cols_order]

        try:
            # Extract probability
            prediction_prob = model.predict_proba(features)[:, 1][0]
            
            # Display results with prominent styling
            st.markdown(f"<h3 style='text-align: center; color: #1f77b4;'>Estimated Probability: {prediction_prob:.1%}</h3>", unsafe_allow_html=True)
            st.progress(float(prediction_prob))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Risk interpretation
            cutoff = 0.50
            if prediction_prob >= cutoff:
                st.error('⚠️ **High Risk Profile:**\n\nThe model indicates an elevated risk of concurrent Coronary Heart Disease. Closer clinical monitoring may be warranted.')
            else:
                st.success('✅ **Low Risk Profile:**\n\nThe model indicates a lower risk of concurrent Coronary Heart Disease. Routine standard-of-care follow-up is recommended.')
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            
    else:
        # 初始状态下的提示语
        st.info("👈 Please fill in the patient characteristics on the left and click **'Calculate CHD Risk'** to generate the report.")
