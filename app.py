import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("reba_model.pkl")

st.title("REBA Score Predictor")
st.write("Enter joint angles to get a predicted REBA score")

# Input fields
neck_flex = st.number_input("Neck Flexion (°)", -90.0, 90.0, 0.0)
neck_lat  = st.number_input("Neck Lateral (°)", -90.0, 90.0, 0.0)
trunk_flex = st.number_input("Trunk Flexion (°)", -90.0, 120.0, 0.0)
trunk_lat = st.number_input("Trunk Lateral (°)", -90.0, 90.0, 0.0)
shoulder_flex = st.number_input("Shoulder Flexion (°)", -90.0, 180.0, 0.0)
elbow_flex = st.number_input("Elbow Flexion (°)", 0.0, 180.0, 90.0)
wrist_flex = st.number_input("Wrist Flexion (°)", -90.0, 90.0, 0.0)
wrist_dev = st.number_input("Wrist Deviation (°)", -45.0, 45.0, 0.0)
knee_left = st.number_input("Left Knee Flexion (°)", 0.0, 150.0, 0.0)
knee_right = st.number_input("Right Knee Flexion (°)", 0.0, 150.0, 0.0)

if st.button("Predict REBA Score"):
    features = np.array([[neck_flex, neck_lat, trunk_flex, trunk_lat,
                          shoulder_flex, elbow_flex, wrist_flex,
                          wrist_dev, knee_left, knee_right]])
    pred = model.predict(features)[0]
    st.success(f"Predicted REBA Score: {int(pred)}")
