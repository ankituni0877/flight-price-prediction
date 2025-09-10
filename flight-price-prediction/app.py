import streamlit as st
import joblib
import pandas as pd
from src.data_processing import engineer_features, get_feature_columns

st.title("✈️ Flight Fare Prediction App")

model = joblib.load("models/flight_fare_rf.pkl")

st.subheader("Enter Flight Details")

airline = st.selectbox("Airline", ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Vistara"])
source = st.selectbox("Source", ["Delhi", "Kolkata", "Mumbai", "Chennai", "Banglore"])
destination = st.selectbox("Destination", ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata", "Banglore"])
date = st.date_input("Journey Date")
dep_time = st.time_input("Departure Time")
arr_time = st.time_input("Arrival Time")
duration = st.text_input("Duration (e.g. 2h 30m)")
stops = st.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops"])
info = st.selectbox("Additional Info", ["No info", "In-flight meal not included", "No check-in baggage included"])

if st.button("Predict Fare"):
    row = {
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Date_of_Journey": date.strftime("%d/%m/%Y"),
        "Dep_Time": dep_time.strftime("%H:%M"),
        "Arrival_Time": arr_time.strftime("%H:%M"),
        "Duration": duration,
        "Total_Stops": stops,
        "Additional_Info": info
    }

    df = pd.DataFrame([row])
    df = engineer_features(df)
    cat_cols, num_cols = get_feature_columns(df)
    df = df[cat_cols + num_cols]
    pred = model.predict(df)[0]

    st.success(f"Estimated Fare: ₹{pred:,.2f}")
