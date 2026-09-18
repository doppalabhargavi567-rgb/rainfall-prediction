import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("decision_tree_model.pkl")

st.title("🌦️ India Weather Rainfall Prediction")
st.write("Enter the weather details to predict rainfall.")

# Get the feature names used by the model
features = model.feature_names_in_

input_data = {}

for feature in features:
    input_data[feature] = st.number_input(
        f"Enter {feature}",
        value=0.0
    )

# Prediction button
if st.button("Predict Rainfall"):

    new_data = pd.DataFrame([input_data])

    prediction = model.predict(new_data)

    if prediction[0] == 1:
        st.success("🌧️ Prediction: Rain")
    else:
        st.write("Prediction: No Rain")


st.title("India Weather Rainfall Prediction")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title("Rainfall Prediction - Graphs")

# Load dataset
df = pd.read_excel("india_weather_rainfall_data.xlsx")

# Show dataset
st.dataframe(df.head())

# 1. Rainfall by Month
st.subheader("Rainfall by Month")
m = df.groupby("month")["rainfall"].mean()
st.bar_chart(m)

# 2. Rainfall by Season
st.subheader("Rainfall by Season")
s = df.groupby("season")["rainfall"].mean()
st.bar_chart(s)

# 3. Rainfall by State
st.subheader("Rainfall by State")
state = df.groupby("state")["rainfall"].mean().sort_values(ascending=False).head(10)
st.bar_chart(state)

# 4. Temperature vs Rainfall
st.subheader("Temperature vs Rainfall")
fig, ax = plt.subplots()
ax.scatter(df["avg_temp"], df["rainfall"], alpha=0.3)
ax.set_xlabel("Average Temperature")
ax.set_ylabel("Rainfall")
st.pyplot(fig)

# 5. Rainfall Distribution
st.subheader("Rainfall Distribution")
fig, ax = plt.subplots()
ax.hist(df["rainfall"].dropna(), bins=30)
ax.set_xlabel("Rainfall")
ax.set_ylabel("Frequency")
st.pyplot(fig)
