import streamlit as st
import numpy as np
from datetime import datetime

from src.features.build_features import build_features
from src.models.wide_deep import load_model

#page_config
st.set_page_config(
    page_title="Dynamic Ride Pricing",
    layout="centered"
)

st.title("🚕 Dynamic Ride Pricing System")
st.write("Wide & Deep Neural Network – Fare Prediction Demo")

#load_model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()
st.success("✅ Model loaded successfully")


if "pickup_date" not in st.session_state:
    st.session_state.pickup_date = datetime(2024, 10, 15).date()

if "pickup_time" not in st.session_state:
    st.session_state.pickup_time = datetime.strptime("10:30", "%H:%M").time()

#input
st.subheader("🧾 Ride Details")

pickup_date = st.date_input(
    "Pickup date",
    value=st.session_state.pickup_date,
    key="pickup_date"
)

pickup_time = st.time_input(
    "Pickup time",
    value=st.session_state.pickup_time,
    key="pickup_time"
)

pickup_lat = st.number_input(
    "Pickup latitude",
    value=40.7128,
    format="%.6f"
)

pickup_lon = st.number_input(
    "Pickup longitude",
    value=-74.0060,
    format="%.6f"
)

dropoff_lat = st.number_input(
    "Dropoff latitude",
    value=40.7306,
    format="%.6f"
)

dropoff_lon = st.number_input(
    "Dropoff longitude",
    value=-73.9352,
    format="%.6f"
)

passenger_count = st.selectbox(
    "Passenger count", [1, 2, 3, 4, 5, 6]
)

#prediction
if st.button("💰 Predict Fare"):
    wide_feats, deep_feats, distance_km = build_features(
        pickup_date,
        pickup_time,
        pickup_lat,
        pickup_lon,
        dropoff_lat,
        dropoff_lon,
        passenger_count
    )

    if distance_km <= 0:
        st.error("Invalid trip distance")
        st.stop()

    wide_input = np.array([wide_feats], dtype=np.float32)
    deep_input = np.array([deep_feats], dtype=np.float32)

    prediction = model.predict([wide_input, deep_input])
    fare = float(prediction[0][0])

    st.success(f"💵 Estimated Fare: **{fare:.2f}**")
