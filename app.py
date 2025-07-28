import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

# Initialize LabelEncoders for each categorical feature
# Note: In production, you should load pre-fitted encoders
workclass_encoder = LabelEncoder()
marital_status_encoder = LabelEncoder()
occupation_encoder = LabelEncoder()
relationship_encoder = LabelEncoder()
race_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
native_country_encoder = LabelEncoder()

# Fit encoders with expected categories (should match training data)
workclass_categories = ["Private", "State-gov", "Federal-gov", "Self-emp-not-inc",
                       "Self-emp-inc", "Local-gov", "Without-pay", "Never-worked"]
workclass_encoder.fit(workclass_categories)

marital_status_categories = [ "Never-married","Married-civ-spouse", "Divorced",
                            "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
marital_status_encoder.fit(marital_status_categories)

occupation_categories = [  "Machine-op-inspct","Tech-support", "Craft-repair", "Other-service", "Sales",
                        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                       "Adm-clerical", "Farming-fishing",
                        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
occupation_encoder.fit(occupation_categories)

relationship_categories = [ "Unmarried","Husband", "Not-in-family", "Wife", "Own-child",
                           "Other-relative"]
relationship_encoder.fit(relationship_categories)


gender_categories = ["Male", "Female"]
gender_encoder.fit(gender_categories)

native_country_categories = ["India","United-States", "Mexico", "Philippines", "Germany",
                            "Canada", "Puerto-Rico", "El-Salvador",
                            "Cuba", "England", "China", "Other"]
native_country_encoder.fit(native_country_categories)

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Personal Information
age = st.sidebar.slider("Age", 18, 70, 28)
gender = st.sidebar.selectbox("Gender", gender_categories)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
marital_status = st.sidebar.selectbox("Marital Status", marital_status_categories)
relationship = st.sidebar.selectbox("Relationship", relationship_categories)
native_country = st.sidebar.selectbox("Native Country", native_country_categories)

# Employment Details
occupation = st.sidebar.selectbox("Occupation", occupation_categories)
workclass = st.sidebar.selectbox("Workclass", workclass_categories)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)


# Create display DataFrame (categorical values)
display_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education Level': [education],
    'Marital Status': [marital_status],
    'Occupation': [occupation],
    'Relationship': [relationship],
    'Native Country': [native_country],
    'Workclass': [workclass],
    'Hours per week': [hours_per_week],
})

# Create prediction DataFrame (encoded values)
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass_encoder.transform([workclass])[0]],
    'marital-status': [marital_status_encoder.transform([marital_status])[0]],
    'occupation': [occupation_encoder.transform([occupation])[0]],
    'relationship': [relationship_encoder.transform([relationship])[0]],
    'gender': [gender_encoder.transform([gender])[0]],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country_encoder.transform([native_country])[0]]
})

st.write("### 🔎 Input Data ")
st.dataframe(display_df.style.set_properties(**{'text-align': 'left'}))

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")




# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Ensure batch data has the same columns
    required_columns = [
        'age', 'workclass',
        'marital-status', 'occupation', 'relationship',
        'gender','hours-per-week',
        'native-country'
    ]

    if all(col in batch_data.columns for col in required_columns):
        # Create a copy for display
        display_batch = batch_data.copy()

        # Encode batch data for prediction
        batch_data['workclass'] = workclass_encoder.transform(batch_data['workclass'])
        batch_data['marital-status'] = marital_status_encoder.transform(batch_data['marital-status'])
        batch_data['occupation'] = occupation_encoder.transform(batch_data['occupation'])
        batch_data['relationship'] = relationship_encoder.transform(batch_data['relationship'])
        batch_data['gender'] = gender_encoder.transform(batch_data['gender'])
        batch_data['native-country'] = native_country_encoder.transform(batch_data['native-country'])

        batch_preds = model.predict(batch_data[required_columns])
        display_batch['PredictedClass'] = batch_preds

        st.write("✅ Predictions:")
        st.dataframe(display_batch)

        csv = display_batch.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    else:
        st.error(f"Uploaded data is missing required columns. Needed: {', '.join(required_columns)}")




