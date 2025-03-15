import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
from pathlib import Path

def load_model(disease_name):
    """Loads the trained model for the given disease."""
    try:
        return joblib.load(f"best_{disease_name}.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to get required model features with correct case
def get_model_features(model_name):
    """Returns the list of features required by each model"""
    if model_name == "heart_disease":
        # Lowercase feature names for heart disease model
        return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    elif model_name == "diabetes":
        # Capitalized feature names for diabetes model
        return ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    elif model_name == "parkinsons":
        # Lowercase feature names for Parkinson's model
        return ['mdvp:fo(hz)', 'mdvp:fhi(hz)', 'mdvp:flo(hz)', 'mdvp:jitter(%)', 
                'mdvp:jitter(abs)', 'mdvp:rap', 'mdvp:ppq', 'jitter:ddp',
                'mdvp:shimmer', 'mdvp:shimmer(db)', 'shimmer:apq3', 'shimmer:apq5',
                'mdvp:apq', 'shimmer:dda', 'nhr', 'hnr', 'rpde', 'dfa',
                'spread1', 'spread2', 'd2', 'ppe']
    return []

def predict_disease(model, input_data, model_name):
    """Predicts disease risk using the trained model with error handling."""
    if model is None:
        return "Model Not Available"
    
    try:
        # For CatBoost specifically, we need to pass a DataFrame with feature names
        if "catboost" in str(type(model)).lower():
            feature_names = get_model_features(model_name)
            if len(input_data) != len(feature_names):
                st.error(f"Expected {len(feature_names)} features but got {len(input_data)}. Please check model requirements.")
                return "Error: Feature Count Mismatch"
            
            # Create DataFrame with correct feature names
            df = pd.DataFrame([input_data], columns=feature_names)
            prediction = model.predict(df)
        else:
            # For sklearn and other models
            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array)
        
        return "Positive" if prediction[0] == 1 else "Negative"
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return f"Error: {str(e)}"

# Create a header with app info
st.set_page_config(page_title="Chronic Disease Prediction", page_icon="🏥", layout="wide")
st.title("Chronic Disease Prediction App")
st.markdown("""
This application uses machine learning to predict the risk of various chronic diseases.
Please select a disease from the sidebar and enter your health information.
""")

# Check if models exist and load them
models_exist = all(Path(f"best_{disease}.pkl").exists() for disease in ["heart_disease", "diabetes", "parkinsons"])

if not models_exist:
    st.warning("""
    ⚠️ Some model files are missing! 
    
    Please make sure all model files (best_heart_disease.pkl, best_diabetes.pkl, best_parkinsons.pkl) 
    are in the same directory as this script.
    
    The app will attempt to continue but predictions may not work properly.
    """)

# Load models with error handling
heart_model = load_model("heart_disease")
diabetes_model = load_model("diabetes")
parkinsons_model = load_model("parkinsons")

# Sidebar navigation
st.sidebar.header("Select Disease")
disease = st.sidebar.selectbox("Choose a disease to check", ["Heart Disease", "Diabetes", "Parkinson's"])

# Heart Disease prediction section
if disease == "Heart Disease":
    st.header("Heart Disease Risk Assessment")
    st.subheader("Enter your health details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=40, step=1)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        sex_value = 1 if sex == "Male" else 0
        cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        cp_value = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120, step=1)
        chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
        fbs_value = 1 if fbs == "Yes" else 0
    
    with col2:
        restecg = st.selectbox("Resting ECG Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        restecg_value = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, step=1)
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
        exang_value = 1 if exang == "Yes" else 0
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
        slope_value = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
        ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0, step=1)
        thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])
        thal_value = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}[thal]
    
    input_features = [age, sex_value, cp_value, trestbps, chol, fbs_value, 
                     restecg_value, thalach, exang_value, oldpeak, 
                     slope_value, ca, thal_value]
    
    if st.button("Predict Heart Disease Risk"):
        with st.spinner("Calculating..."):
            result = predict_disease(heart_model, input_features, "heart_disease")
            
            if "Error" in result:
                st.error(f"Prediction failed: {result}")
            else:
                if result == "Positive":
                    st.error(f"⚠️ Prediction: {result} - Higher risk of heart disease detected")
                    st.info("Please consult with a healthcare professional for proper evaluation.")
                else:
                    st.success(f"✅ Prediction: {result} - Lower risk of heart disease")
                    st.info("Continue maintaining a healthy lifestyle!")

# Diabetes prediction section
elif disease == "Diabetes":
    st.header("Diabetes Risk Assessment")
    st.subheader("Enter your health details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
        glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=120, step=1)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=21, max_value=100, value=35, step=1)
    
    # Include ALL the required features in the correct order
    input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    
    if st.button("Predict Diabetes Risk"):
        with st.spinner("Calculating..."):
            result = predict_disease(diabetes_model, input_features, "diabetes")
            
            if "Error" in result:
                st.error(f"Prediction failed: {result}")
            else:
                if result == "Positive":
                    st.error(f"⚠️ Prediction: {result} - Higher risk of diabetes detected")
                    st.info("Please consult with a healthcare professional for proper evaluation.")
                else:
                    st.success(f"✅ Prediction: {result} - Lower risk of diabetes")
                    st.info("Continue maintaining a healthy lifestyle!")

# Parkinson's prediction section
elif disease == "Parkinson's":
    st.header("Parkinson's Disease Risk Assessment")
    st.subheader("Enter voice recording measurements")
    
    # Create a more manageable UI with tabs for the many Parkinson's features
    tab1, tab2, tab3 = st.tabs(["Frequency Measures", "Jitter Measures", "Other Measures"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            mdvp_fo = st.number_input("MDVP:Fo(Hz) - Average vocal fundamental frequency", 
                                     min_value=80.0, max_value=300.0, value=120.0, step=0.1)
            mdvp_fhi = st.number_input("MDVP:Fhi(Hz) - Maximum vocal fundamental frequency", 
                                      min_value=80.0, max_value=600.0, value=150.0, step=0.1)
        with col2:
            mdvp_flo = st.number_input("MDVP:Flo(Hz) - Minimum vocal fundamental frequency", 
                                      min_value=50.0, max_value=300.0, value=100.0, step=0.1)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            mdvp_jitter = st.number_input("MDVP:Jitter(%) - Frequency variation percentage", 
                                         min_value=0.0, max_value=1.0, value=0.006, step=0.001, format="%.6f")
            mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs) - Absolute jitter", 
                                             min_value=0.0, max_value=1.0, value=0.00004, step=0.00001, format="%.6f")
            mdvp_rap = st.number_input("MDVP:RAP - Relative amplitude perturbation", 
                                      min_value=0.0, max_value=1.0, value=0.003, step=0.001, format="%.6f")
            mdvp_ppq = st.number_input("MDVP:PPQ - Five-point period perturbation quotient", 
                                      min_value=0.0, max_value=1.0, value=0.003, step=0.001, format="%.6f")
        with col2:
            jitter_ddp = st.number_input("Jitter:DDP - Average perturbation measure", 
                                        min_value=0.0, max_value=1.0, value=0.009, step=0.001, format="%.6f")
            mdvp_shimmer = st.number_input("MDVP:Shimmer - Amplitude variation", 
                                          min_value=0.0, max_value=1.0, value=0.04, step=0.01)
            mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB) - Amplitude variation in dB", 
                                             min_value=0.0, max_value=3.0, value=0.3, step=0.1)
            shimmer_apq3 = st.number_input("Shimmer:APQ3 - Three-point amplitude quotient", 
                                          min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            shimmer_apq5 = st.number_input("Shimmer:APQ5 - Five-point amplitude quotient", 
                                          min_value=0.0, max_value=1.0, value=0.03, step=0.01)
            mdvp_apq = st.number_input("MDVP:APQ - Amplitude perturbation quotient", 
                                      min_value=0.0, max_value=1.0, value=0.03, step=0.01)
            shimmer_dda = st.number_input("Shimmer:DDA - Amplitude differences", 
                                         min_value=0.0, max_value=1.0, value=0.06, step=0.01)
            nhr = st.number_input("NHR - Noise to harmonic ratio", 
                                 min_value=0.0, max_value=1.0, value=0.02, step=0.01)
            hnr = st.number_input("HNR - Harmonic to noise ratio", 
                                 min_value=0.0, max_value=40.0, value=20.0, step=0.1)
        with col2:
            rpde = st.number_input("RPDE - Recurrence period density entropy", 
                                  min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            dfa = st.number_input("DFA - Signal fractal scaling exponent", 
                                 min_value=0.0, max_value=3.0, value=0.7, step=0.01)
            spread1 = st.number_input("spread1 - Nonlinear measure of fundamental frequency variation", 
                                     min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            spread2 = st.number_input("spread2 - Second nonlinear measure", 
                                     min_value=0.0, max_value=1.0, value=0.2, step=0.01)
            d2 = st.number_input("D2 - Correlation dimension", 
                                min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            ppe = st.number_input("PPE - Pitch period entropy", 
                                 min_value=0.0, max_value=3.0, value=0.2, step=0.01)
    
    # Include ALL required features for Parkinson's prediction
    input_features = [
        mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter_abs, mdvp_rap, mdvp_ppq,
        jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5,
        mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
    ]
    
    if st.button("Predict Parkinson's Risk"):
        with st.spinner("Calculating..."):
            result = predict_disease(parkinsons_model, input_features, "parkinsons")
            
            if "Error" in result:
                st.error(f"Prediction failed: {result}")
            else:
                if result == "Positive":
                    st.error(f"⚠️ Prediction: {result} - Higher risk of Parkinson's detected")
                    st.info("Please consult with a healthcare professional for proper evaluation.")
                else:
                    st.success(f"✅ Prediction: {result} - Lower risk of Parkinson's")
                    st.info("The voice analysis indicates lower likelihood of Parkinson's disease.")

# Add disclaimer at the bottom
st.markdown("---")
st.markdown("""
**Disclaimer**: This app is for educational purposes only and should not be used as a substitute 
for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
or other qualified health provider with any questions regarding a medical condition.
""")

# Display model information
with st.expander("Technical Information"):
    st.markdown("""
    ### Model Information
    
    This application uses machine learning models to predict disease risk.
    
    - **Heart Disease**: Model predicts coronary heart disease risk based on clinical factors
    - **Diabetes**: Model predicts diabetes risk based on the Pima Indians Diabetes Dataset
    - **Parkinson's**: Model predicts Parkinson's disease risk based on voice recording measurements
    
    Please note that these predictions are statistical in nature and not definitive diagnoses.
    """)