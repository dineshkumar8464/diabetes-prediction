import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load model and scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load images (use your own image paths or URLs)
doctor_img = Image.open("C:\\Users\\dines\\OneDrive\\Desktop\\Projects\\diabetes\\doctor image for positive result.jpg")  # Doctor image for positive result
cheers_img = Image.open("C:\\Users\\dines\\OneDrive\\Desktop\\Projects\\diabetes\\959-9595288_clip-art-royalty-free-download-icon-three-cheering.png")  # Cheers image for negative result

# Title and intro

st.sidebar.markdown("""
## Welcome to the Diabetes Risk Predictor

This simple and intelligent tool helps you understand your **risk of having diabetes** based on your basic health information.

> **What is Diabetes?**  
Diabetes is a common health condition where your blood sugar (glucose) levels become too high. Over time, if untreated, it can affect your heart, kidneys, vision, and overall well-being.  
But don't worry — early awareness can make a huge difference!

---

### How does this app help?
By entering a few details like your age, glucose levels, and BMI, this app uses a smart machine learning model to check your diabetes risk and offer helpful suggestions.

>  **Note**: This is *not* a medical diagnosis, but it can guide you to make better health choices or seek medical advice if needed.
""")

st.title("Diabetes Prediction App")
st.markdown("## About This Application\nUnderstand your diabetes risk easily using your basic health data.")

import streamlit as st
import pandas as pd

# Function to validate text input without stopping UI
def get_validated_input(label, default, dtype, min_val, max_val):
    raw = st.text_input(f"{label} (Range: {min_val}–{max_val})", str(default))
    value = None

    if raw.strip():  # Check if something was entered
        try:
            val = dtype(raw)
            if val < min_val or val > max_val:
                st.warning(f" {label} must be between {min_val} and {max_val}. You entered {val}.")
            else:
                value = val
        except:
            st.warning(f" Please enter a valid {dtype.__name__} value for {label}.")
    else:
        st.info(f" {label} is required.")

    return value

# Main feature input function
def user_input_features():
    st.subheader("Enter Your Health Data")
    use_sliders = st.toggle("Prefer guided input? Switch to slider mode")

    if not use_sliders:
        st.markdown("Enter values from your health report. Allowed ranges are shown.")

        pregnancies = get_validated_input("Number of Pregnancies", 1, int, 0, 20)
        glucose = get_validated_input("Glucose Level (mg/dL)", 120, float, 44, 200)
        blood_pressure = get_validated_input("Blood Pressure (mm Hg)", 70, float, 24, 122)
        skin_thickness = get_validated_input("Skin Thickness (mm)", 20, float, 7, 99)
        insulin = get_validated_input("Insulin Level (mu U/ml)", 80, float, 15, 846)
        bmi = get_validated_input("BMI (Body Mass Index)", 28.0, float, 18.0, 67.1)
        dpf = get_validated_input("Diabetes Pedigree Function", 0.37, float, 0.078, 2.42)
        age = get_validated_input("Age (years)", 33, int, 21, 81)

    else:
        st.markdown("Use sliders to select approximate values.")

        pregnancies = st.slider('Number of Pregnancies', 0, 20, 1)
        glucose = st.slider('Glucose Level (mg/dL)', 44, 200, 120)
        blood_pressure = st.slider('Blood Pressure (mm Hg)', 24, 122, 70)
        skin_thickness = st.slider('Skin Thickness (mm)', 7, 99, 20)
        insulin = st.slider('Insulin Level (mu U/ml)', 15, 846, 80)
        bmi = st.slider('BMI (Body Mass Index)', 18.0, 67.1, 28.0)
        dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.37)
        age = st.slider('Age (years)', 21, 81, 33)

    # Don’t proceed if using text inputs and any value is None
    if not use_sliders and None in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]:
        st.info("Please correct the invalid inputs to proceed.")
        return None  # Return nothing — don’t proceed to prediction

    # Prepare final DataFrame for prediction
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Show input summary
st.subheader("Summary of Your Input")
st.write(input_df)

# Scale inputs
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("Predict Diabetes"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][prediction]

    # Output section
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("High Risk: The model predicts you may have diabetes.")
        st.image(doctor_img, width=200)
        st.markdown("""
        ### What to do if positive:
        - Consult a healthcare professional for detailed testing.
        - Maintain a balanced diet and monitor blood sugar levels.
        - Exercise regularly and manage weight.
        - Avoid sugary foods and smoking.
        """)
    else:
        st.success("Low Risk: The model predicts you likely do NOT have diabetes.")
        st.image(cheers_img, width=200)
        st.markdown("""
        ### Suggestions if negative:
        - Maintain a healthy lifestyle to stay diabetes-free.
        - Regular exercise and balanced diet are key.
        - Get routine checkups for early detection.
        - Avoid excessive sugar and processed foods.
        """)

    # Confidence score
    st.write(f"Model confidence: {proba*100:.2f}%")

    # Detailed explanation of criteria (simple)
    st.markdown("""
    ---
    ### How does the model decide?
    It looks at:
    - **Glucose level**: High levels often indicate diabetes.
    - **Blood Pressure & BMI**: Important factors influencing diabetes risk.
    - **Age & Family History** (Diabetes Pedigree Function).
    - **Other health parameters** like insulin and skin thickness.
    """)

    st.markdown("Thank you for using the app. Remember, this is a prediction and not a diagnosis.")

# Example section for users
st.sidebar.markdown("Example Inputs")
st.sidebar.markdown("""
Try these example cases to see how the model reacts:

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DPF  | Age |
|-------------|---------|---------------|---------------|---------|------|------|-----|
| 2           | 140     | 80            | 30            | 100     | 35.0 | 0.5  | 45  |
| 0           | 90      | 60            | 20            | 80      | 22.0 | 0.3  | 25  |
""")
