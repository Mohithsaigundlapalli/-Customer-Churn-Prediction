import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure")
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

if st.button("Predict"):
    data = np.array([[tenure, monthly, total]])
    result = model.predict(data)

    if result[0] == 1:
        st.error("Customer will churn ")
    else:
        st.success("Customer will stay ")
