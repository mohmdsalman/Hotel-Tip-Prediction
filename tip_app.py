import pandas as pd
import streamlit as st
import pickle


st.title("Tip prediction")

total_bill = st.number_input("total_bill", max_value=100.0, min_value=0.0, value=10.0, step=0.1)
time = st.selectbox("time", ["Dinner", "Lunch"])
size = st.number_input("size", max_value=20, min_value=1, value=2)

input_data = {"total_bill":total_bill,
              "time":time,
              "size":size}

data = pd.DataFrame([input_data])

encoded_time = {"Dinner" : 0, "Lunch" : 1}

data["time"] = data["time"].map(encoded_time)

df = pd.read_csv("features.csv")
columns_list = [col for col in df.columns if col != 'Unnamed: 0']
data = data.reindex(columns=columns_list, fill_value=0)

model = pickle.load(open("le_model.pkl", "rb"))

prediction = model.predict(data)
pred = prediction[-1]

if st.button("SUBMIT"):
    st.success(f"The predicted tip is : ${pred}")
    