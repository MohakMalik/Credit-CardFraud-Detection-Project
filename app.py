import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details below to predict whether the transaction is fraudulent.")


# --------------------------------------------------
# Load saved model bundle
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")


model_bundle = load_model()

rf = model_bundle["model"]
scaler = model_bundle["scaler"]
encoder = model_bundle["encoder"]
job_freq = model_bundle["job_freq"]
merchant_freq = model_bundle["merchant_freq"]
feature_columns = model_bundle["feature_columns"]


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

st.subheader("Transaction Details")

amt = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

gender = st.selectbox(
    "Gender",
    ["M", "F"]
)

category = st.selectbox(
    "Transaction Category",
    [
        "grocery_pos",
        "shopping_pos",
        "gas_transport",
        "misc_pos",
        "shopping_net",
        "grocery_net",
        "misc_net",
        "entertainment",
        "food_dining",
        "personal_care",
        "health_fitness",
        "home",
        "kids_pets",
        "travel"
    ]
)

job = st.text_input(
    "Job",
    value="Software engineer"
)

merchant = st.text_input(
    "Merchant",
    value="fraud_Rippin, Kub and Mann"
)

lat = st.number_input(
    "Customer Latitude",
    value=40.0
)

long = st.number_input(
    "Customer Longitude",
    value=-75.0
)

merch_lat = st.number_input(
    "Merchant Latitude",
    value=40.0
)

merch_long = st.number_input(
    "Merchant Longitude",
    value=-75.0
)

dob = st.date_input(
    "Date of Birth",
    value=pd.Timestamp("1990-01-01")
)

transaction_datetime = st.date_input(
    "Transaction Date",
    value=pd.Timestamp.today()
)

transaction_time = st.time_input(
    "Transaction Time"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Fraud"):

    # Convert date/time
    transaction_date = pd.Timestamp(transaction_datetime)
    dob = pd.Timestamp(dob)

    # Calculate customer age
    customer_age = round(
        (transaction_date - dob).days / 365.25
    )

    # Extract transaction hour
    transaction_hour = transaction_time.hour

    # Gender mapping used in notebook
    gender_value = 0 if gender == "M" else 1

    # Frequency encoding
    job_frequency = job_freq.get(job, 0)
    merchant_frequency = merchant_freq.get(merchant, 0)

    # Create input dataframe
    input_data = pd.DataFrame({
        "amt": [amt],
        "gender": [gender_value],
        "lat": [lat],
        "long": [long],
        "merch_lat": [merch_lat],
        "merch_long": [merch_long],
        "customer_age": [customer_age],
        "transaction_hour": [transaction_hour],
        "job_frequency": [job_frequency],
        "merchant_frequency": [merchant_frequency],
        "category": [category]
    })


    # --------------------------------------------------
    # One-hot encode category
    # --------------------------------------------------

    encoded = encoder.transform(input_data[["category"]])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["category"]),
        index=input_data.index
    )

    input_data = pd.concat(
        [
            input_data.drop(columns=["category"]),
            encoded_df
        ],
        axis=1
    )


    # --------------------------------------------------
    # Scale numerical features
    # --------------------------------------------------

    scaling_features = [
        "amt",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "customer_age",
        "transaction_hour",
        "job_frequency",
        "merchant_frequency"
    ]

    input_data[scaling_features] = scaler.transform(
        input_data[scaling_features]
    )


    # --------------------------------------------------
    # Make sure feature order matches training
    # --------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = rf.predict(input_data)[0]

    probability = rf.predict_proba(input_data)[0][1]


    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")

    else:
        st.success("✅ Legitimate Transaction")

    st.write(
        f"Fraud Probability: **{probability:.2%}**"
    )