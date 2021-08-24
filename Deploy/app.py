import streamlit as st
import joblib
import pandas as pd
import pickle


def inference(row, model, feat_cols):
    df = pd.DataFrame([row], columns=feat_cols)
    print(float(model.predict_proba(df)[0][1]))
    if float(model.predict_proba(df)[0][1]) < 0.5:
        return "This is a healthy person!"
    else:
        return "This person has high chances of having diabetics!"


st.title("Diabetes Prediction App")
st.write(
    "Please fill in the details of the person under consideration in the left sidebar and click on the button below!"
)

age = st.sidebar.slider("Age in Years", 1, 150, 25, 1)
systolic = st.sidebar.slider("Systolic blood pressure", 90, 210, 125, 1)
diastolic = st.sidebar.slider("Diastolic blood pressure", 60, 130, 80, 1)
hdl = st.sidebar.slider("High-density lipoprotein", 10.0, 100.0, 65.5, 0.05)
ldl = st.sidebar.slider("Low-density lipoprotein", 40.0, 210.0, 85.5, 0.05)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 31.4, 0.05)

row = [systolic, diastolic, hdl, ldl, bmi, age]

if st.button("Find Health Status"):
    feat_cols = ["systolic", "diastolic", "hdl", "ldl", "bmi", "age"]
    model = pickle.load(open(r"data/model.pkl", "rb"))
    result = inference(row, model, feat_cols)
    st.write(result)
