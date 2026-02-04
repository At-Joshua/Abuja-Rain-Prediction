import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Page config
st.set_page_config(
    page_title="Abuja Rain Prediction 🌧️",
    page_icon="🌦️",
    layout="centered"
)

col1, col2 = st.columns((1, 1))

with col1:
    st.image("https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3OGo2OGlkZmoybmRwczlidWM0ZnhhY3lmZHRvOXRoazQ0OHdyZDJ0OCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/PR5vbMZ6zDaR9EugXY/giphy.gif")

with col2:

    st.image(
        "https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy.gif",
        use_container_width=True
    )


st.title("🌧️ Abuja Rain Prediction App")
st.markdown("""
### About This App

This Weather Detection App analyzes historical weather data and uses a machine learning model to predict whether it will rain on a given day.

The app is built with **Python**, **scikit-learn**, and **Streamlit**, and is designed to demonstrate the practical application of data analytics and machine learning in weather prediction.

⚠️ *This application is for educational purposes only and should not replace official weather forecasts.*
""")

st.write("--Predict whether it will rain on a selected day using historical weather data.--")


@st.cache_data
def load_weather_data():
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=9.0765"
        "&longitude=7.3986"
        "&start_date=2015-01-01"
        "&end_date=2024-12-31"
        "&daily=temperature_2m_mean,"
        "relative_humidity_2m_mean,"
        "precipitation_sum,"
        "windspeed_10m_max"
        "&timezone=Africa/Lagos"
    )

    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp": data["daily"]["temperature_2m_mean"],
        "humidity": data["daily"]["relative_humidity_2m_mean"],
        "precipitation": data["daily"]["precipitation_sum"],
        "wind_speed": data["daily"]["windspeed_10m_max"]
    })

    # Feature engineering
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Target variable
    df["rain"] = (df["precipitation"] > 0).astype(int)

    return df

df = load_weather_data()

# Train model
features = ["temp", "humidity", "wind_speed", "year", "month", "day"]
X = df[features]
y = df["rain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Saved model
joblib.dump(model, "rain_prediction_model.joblib")

# Input Section

st.subheader("📅 Please Select a Date")

selected_date = st.date_input("Please Choose a day")

input_year = selected_date.year
input_month = selected_date.month
input_day = selected_date.day

# Use historical averages for weather inputs
avg_temp = df["temp"].mean()
avg_humidity = df["humidity"].mean()
avg_wind = df["wind_speed"].mean()

input_data = pd.DataFrame([[
    avg_temp,
    avg_humidity,
    avg_wind,
    input_year,
    input_month,
    input_day
]], columns=features)

# Prediction

if st.button("🌦️ Predict Rain"):
    rain_prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]



    if prediction == 1:
        st.image(
            "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3OGo2OGlkZmoybmRwczlidWM0ZnhhY3lmZHRvOXRoazQ0OHdyZDJ0OCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/0NVn5SSlX85kUpIJFi/giphy.gif",
            use_container_width=True
        )
        st.success(f"🌧️ It is likely to rain!\n\nProbability: **{rain_prob:.2%}**")
    else:
        st.image("https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3aDZkbDVqbjF5eThyMG1jcDh6bXVlZ3Q1bHh6ZGIwYjl1ZTkzNzM2OCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/dvfr9hXPvjrBOgBTE4/giphy.gif")
        st.info(f"☀️ No rain expected.\n\nProbability of rain: **{rain_prob:.2%}**")
# Footer
st.markdown("---")
st.caption(" This App is Built with Open-Meteo API, Random Forest & Streamlit By JOSH🚀")
