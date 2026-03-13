import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="COPD-CHD Risk Predictor", page_icon="🩺")
st.title("🩺 CHD Risk Prediction Model for Elderly COPD Patients")
st.markdown("""
**Description:** This web-based interactive calculator predicts the risk of concurrent Coronary Heart Disease (CHD) in elderly patients with Chronic Obstructive Pulmonary Disease (COPD) using a Machine Learning (Random Forest) approach.
***Disclaimer:** This tool is strictly for academic research and validation purposes. It should not be utilized as a substitute for professional clinical judgment or direct patient care.*
""")

# 2. Load the Random Forest Model
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model. Please ensure 'rf_model.pkl' is in the same directory. Error: {e}")
    st.stop()

# 3. Sidebar for Patient Characteristics Input
st.sidebar.header("Patient Characteristics")

def user_input_features():
    # Categorical variables
    gender_option = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    gender = 1 if gender_option == 'Male' else 0

    # Continuous variables 
    age = st.sidebar.number_input('Age (years)', min_value=40, max_value=120, value=75)
    ua = st.sidebar.number_input('Uric Acid (UA, μmol/L)', min_value=0.0, max_value=1000.0, value=350.0)
    hs_crp = st.sidebar.number_input('High-sensitivity CRP (hs-CRP, mg/L)', min_value=0.0, max_value=200.0, value=5.0)
    glu = st.sidebar.number_input('Fasting Glucose (Glu, mmol/L)', min_value=0.0, max_value=50.0, value=5.5)
    bun = st.sidebar.number_input('Blood Urea Nitrogen (BUN, mmol/L)', min_value=0.0, max_value=100.0, value=6.0)
    hs_ctni = st.sidebar.number_input('High-sensitivity cTnI (hs-cTnI, pg/mL)', min_value=0.0, max_value=10000.0, value=15.0)
    tt = st.sidebar.number_input('Thrombin Time (TT, s)', min_value=0.0, max_value=100.0, value=18.0)

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
    
    # Ensure column order matches exactly
    cols_order = ['GENDER', 'UA', 'hs_CRP', 'Glu', 'AGE', 'BUN', 'hs_cTnI', 'TT']
    features = features[cols_order]
    
    return features

input_df = user_input_features()

# 4. Prediction Button and Result Display
st.divider()
if st.button('Predict CHD Risk'):
    try:
        # Extract the probability of class 1 (CHD)
        prediction_prob = model.predict_proba(input_df)[:, 1][0]
        
        st.subheader('Prediction Result')
        st.metric(label="Estimated Probability of Concurrent CHD", value=f"{prediction_prob:.1%}")
        
        # Visual progress bar
        st.progress(float(prediction_prob))
        
        # Risk assessment interpretation
        cutoff = 0.50
        if prediction_prob >= cutoff:
            st.error('⚠️ **High Risk Profile:** The model indicates an elevated risk of concurrent Coronary Heart Disease for this patient. Closer clinical monitoring and further cardiovascular assessment may be warranted.')
        else:
            st.success('✅ **Low Risk Profile:** The model indicates a lower risk of concurrent Coronary Heart Disease. Routine standard-of-care follow-up is recommended.')

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.warning("Please verify that the feature names and data types match the training environment exactly.")
