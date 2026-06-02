import streamlit as st
import pandas as pd
import pickle

df = pd.read_csv("data/hospital_prices.csv")

with open("price_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Hospital Price Fairness Analyzer")
st.write("Check whether hospital service prices are fair using machine learning")

service = st.selectbox("Select Service", df["service"].unique())
city = st.selectbox("Select City", df["city"].unique())
rating = st.slider("Hospital Rating", 1.0, 5.0, 4.0)
wait_time = st.slider("Waiting Time (minutes)", 0, 120, 30)

input_data = pd.DataFrame({
    "wait_time": [wait_time],
    "rating": [rating],
    "city_Delhi": [1 if city == "Delhi" else 0],
    "city_Mumbai": [1 if city == "Mumbai" else 0],
    "service_X-Ray": [1 if service == "X-Ray" else 0],
    "service_MRI": [1 if service == "MRI" else 0],
    "service_Blood Test": [1 if service == "Blood Test" else 0]
})

input_data = input_data[model.feature_names_in_]


predicted_price = model.predict(input_data)[0]
st.subheader("Predicted Fair Price")
st.write(f"Rs {int(predicted_price)}")

st.subheader("Price Fairness")

actual_price = st.number_input(
    "Enter Actual Price Charged by Hospital",
    min_value=0
)

if actual_price > 0:
    fairness_score = actual_price / predicted_price
    if fairness_score > 1.2:
        st.error("Overpriced")
    elif fairness_score < 0.8:
        st.warning("Underpriced")
    else:
        st.success("Fair Price")
