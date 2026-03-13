import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration (设置为宽屏或者居中，这里保持默认居中显得紧凑)
st.set_page_config(page_title="COPD-CHD Risk Predictor", page_icon="🩺", layout="centered")

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

# 4. Main Input Section (使用表单和多列排版)
with st.form("prediction_form"):
    st.subheader("Patient Clinical Characteristics")
    
    # 第一行：人口学与基本指标
    col1, col2, col3 = st.columns(3)
    with col1:
        gender_option = st.selectbox('Gender', ['Male', 'Female'])
        gender = 1 if gender_option == 'Male' else 0
    with col2:
        age = st.number_input('Age (years)', min_value=40, max_value=120, value=75)
    with col3:
        glu = st.number_input('Fasting Glucose (mmol/L)', min_value=0.0, max_value=50.0, value=5.5)

    st.markdown("<br>", unsafe_allow_html=True) # 增加一点垂直间距
    
    # 第二行：实验室检查指标 1
    col4, col5 = st.columns(2)
    with col4:
        ua = st.number_input('Uric Acid (μmol/L)', min_value=0.0, max_value=1000.0, value=350.0)
    with col5:
        bun = st.number_input('Blood Urea Nitrogen (mmol/L)', min_value=0.0, max_value=100.0, value=6.0)

    st.markdown("<br>", unsafe_allow_html=True)

    # 第三行：实验室检查指标 2 (心肌与炎症标志物)
    col6, col7, col8 = st.columns(3)
    with col6:
        hs_crp = st.number_input('hs-CRP (mg/L)', min_value=0.0, max_value=200.0, value=5.0)
    with col7:
        hs_ctni = st.number_input('hs-cTnI (pg/mL)', min_value=0.0, max_value=10000.0, value=15.0)
    with col8:
        tt = st.number_input('Thrombin Time (s)', min_value=0.0, max_value=100.0, value=18.0)

    # 提交按钮
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Calculate CHD Risk", use_container_width=True)

# 5. Prediction Logic and Result Display
if submit_button:
    # Ensure keys match exactly with the model's feature names
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
        # Extract the probability of class 1 (CHD)
        prediction_prob = model.predict_proba(features)[:, 1][0]
        
        st.divider()
        st.subheader('Prediction Result')
        
        # 使用多列让结果显示更居中、突出
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Estimated Probability", value=f"{prediction_prob:.1%}")
        
        with res_col2:
            cutoff = 0.50
            if prediction_prob >= cutoff:
                st.error('⚠️ **High Risk Profile:** The model indicates an elevated risk of concurrent Coronary Heart Disease. Closer clinical monitoring may be warranted.')
            else:
                st.success('✅ **Low Risk Profile:** The model indicates a lower risk of concurrent Coronary Heart Disease. Routine standard-of-care follow-up is recommended.')
                
        # 进度条放在最后
        st.progress(float(prediction_prob))

    except Exception as e:
        st.error(f"Prediction Error: {e}")
