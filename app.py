import os
import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# 1. Load Model dan Scaler
model = tf.keras.models.load_model("Model/breast_cancer_pred_model.keras")
scaler = joblib.load("Model/scaler.pkl")

# 2. Application Title
st.title("Breast Cancer Prediction App")
st.write("Use either manual input or upload a CSV file to perform breast cancer prediction.")

# 3. Feature Names
features = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error",
    "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

# 4. Select Input Method
input_method = st.radio(
    "Select Input Method:",
    ["Manual Input", "CSV File"],
    horizontal=True
)

# 5. Manual Input
if input_method == "Manual Input":
    st.subheader("Enter Feature Values")
    input_data = []

    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        if i % 2 == 0:
            with col1:
                value = st.number_input(feature, min_value=0.0, value=0.0, key=feature)
        else:
            with col2:
                value = st.number_input(feature, min_value=0.0, value=0.0, key=feature)
        input_data.append(value)

# 6. CSV File Input
else:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        import pandas as pd
        csv_data = pd.read_csv(uploaded_file)
        
        column_mapping = {
            'radius_mean': 'mean radius', 'texture_mean': 'mean texture', 
            'perimeter_mean': 'mean perimeter', 'area_mean': 'mean area', 
            'smoothness_mean': 'mean smoothness', 'compactness_mean': 'mean compactness', 
            'concavity_mean': 'mean concavity', 'concave points_mean': 'mean concave points', 
            'symmetry_mean': 'mean symmetry', 'fractal_dimension_mean': 'mean fractal dimension',
            'radius_se': 'radius error', 'texture_se': 'texture error', 
            'perimeter_se': 'perimeter error', 'area_se': 'area error', 
            'smoothness_se': 'smoothness error', 'compactness_se': 'compactness error', 
            'concavity_se': 'concavity error', 'concave points_se': 'concave points error', 
            'symmetry_se': 'symmetry error', 'fractal_dimension_se': 'fractal dimension error',
            'radius_worst': 'worst radius', 'texture_worst': 'worst texture', 
            'perimeter_worst': 'worst perimeter', 'area_worst': 'worst area', 
            'smoothness_worst': 'worst smoothness', 'compactness_worst': 'worst compactness', 
            'concavity_worst': 'worst concavity', 'concave points_worst': 'worst concave points', 
            'symmetry_worst': 'worst symmetry', 'fractal_dimension_worst': 'worst fractal dimension'
        }
        
        # Rename columns if they match the Kaggle naming convention
        csv_data = csv_data.rename(columns=column_mapping)

        st.write("Preview of uploaded data:")
        st.dataframe(csv_data.head())

        missing_features = [feature for feature in features if feature not in csv_data.columns]

        if missing_features:
            st.error("The CSV file is missing the following features:")
            st.write(missing_features)
        else:
            st.success(f"CSV file loaded successfully. {len(csv_data)} records found.")

# 7. Run Prediction
st.divider()

if st.button("Run Predict", type="primary"):

    # Manual Input Prediction
    if input_method == "Manual Input":
        input_df = pd.DataFrame([input_data], columns=features)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled, verbose=0)
        probability = prediction[0][0]

        result = "Malignant" if probability >= 0.5 else "Benign"

        st.subheader("Prediction Result")
        if result == "Malignant":
            st.error(f"Prediction: **{result}**")
        else:
            st.success(f"Prediction: **{result}**")

        st.write(f"Prediction probability: **{probability:.4f}**")

else:
        if uploaded_file is None:
            st.warning("Please upload a CSV file first.")
        elif missing_features:
            st.error("Prediction cannot be performed because some required features are missing.")
        else:
            X_new = csv_data[features]
            X_scaled = scaler.transform(X_new)
            predictions = model.predict(X_scaled, verbose=0)

            probabilities = predictions.flatten()
            results = np.where(probabilities >= 0.5, "Malignant", "Benign")

            result_df = csv_data.copy()
            result_df["Prediction Probability"] = probabilities
            result_df["Prediction"] = results

            st.subheader("Prediction Results")
            st.dataframe(result_df)

            malignant_count = np.sum(results == "Malignant")
            benign_count = np.sum(results == "Benign")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Benign", benign_count)
            with col2:
                st.metric("Malignant", malignant_count)

            csv_output = result_df.to_csv(index=False)
            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="breast_cancer_predictions.csv",
                mime="text/csv"
            )
