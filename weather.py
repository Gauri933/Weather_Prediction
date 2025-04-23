import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set page config
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .reportview-container {
        background: #f0f2f6
    }
    </style>
    """, unsafe_allow_html=True)

API_KEY = "b487df3d8bd7a5c686b50d0806a3a460"
UNITS = "metric"

# City selection
st.title("🌤️ Weather Prediction App")
CITY = st.selectbox("Select City", ["Mumbai", "London", "New York", "Tokyo", "Paris"])

def get_weather_data(city, api_key, units="metric"):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units={units}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error(f"Error: {response.status_code}") #streamlit error
        return None

def process_weather_data(data):
    if not data:
        return None

    weather_list = []
    for item in data["list"]:
        timestamp = item["dt"]
        date_time = datetime.fromtimestamp(timestamp)
        temperature = item["main"]["temp"]
        humidity = item["main"]["humidity"]
        pressure = item["main"]["pressure"]
        wind_speed = item["wind"]["speed"]
        description = item["weather"][0]["description"]

        weather_list.append({
            "datetime": date_time,
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "description": description,
        })

    df = pd.DataFrame(weather_list)
    return df

# Create two columns for the layout
col1, col2 = st.columns([2, 1])

weather_data = get_weather_data(CITY, API_KEY, UNITS)

if weather_data:
    weather_df = process_weather_data(weather_data)
    weather_df["hour"] = weather_df["datetime"].dt.hour
    
    with col1:
        st.subheader("📈 Historical Weather Data")
        # Plot temperature over time
        fig_temp = px.line(weather_df, x="datetime", y="temperature",
                          title="Temperature Variation Over Time",
                          labels={"temperature": "Temperature (°C)", "datetime": "Date & Time"})
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Plot other weather metrics
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Scatter(x=weather_df["datetime"], y=weather_df["humidity"], name="Humidity (%)"))
        fig_metrics.add_trace(go.Scatter(x=weather_df["datetime"], y=weather_df["wind_speed"], name="Wind Speed (m/s)"))
        fig_metrics.update_layout(title="Weather Metrics Over Time")
        st.plotly_chart(fig_metrics, use_container_width=True)

    X = weather_df[["temperature", "humidity", "pressure", "wind_speed", "hour"]]
    y = weather_df["temperature"].shift(-1)
    X = X[:-1]
    y = y[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    with col2:
        st.subheader("🎯 Prediction Panel")
        with st.container():
            st.markdown(f"**Model Performance**: MSE = {mse:.4f}")
            
            st.markdown("### 📝 Enter Weather Parameters")
            temperature = st.number_input("Temperature (°C)", value=15.0, help="Current temperature in Celsius")
            humidity = st.number_input("Humidity (%)", value=60, min_value=0, max_value=100, help="Current humidity percentage")
            pressure = st.number_input("Pressure (hPa)", value=1015, help="Current atmospheric pressure")
            wind_speed = st.number_input("Wind Speed (m/s)", value=10.0, min_value=0.0, help="Current wind speed")
            hour = st.slider("Hour of Day", 0, 23, 12, help="Hour of the day (24-hour format)")

            if st.button("🔮 Predict Temperature"):
                new_data = pd.DataFrame({
                    "temperature": [temperature],
                    "humidity": [humidity],
                    "pressure": [pressure],
                    "wind_speed": [wind_speed],
                    "hour": [hour],
                })
                prediction = model.predict(new_data)
                
                st.markdown("### 🌡️ Prediction Result")
                st.markdown(
                    f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h2 style='text-align: center; color: #1f77b4;'>{prediction[0]:.1f}°C</h2>
                        <p style='text-align: center;'>Predicted Temperature</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
else:
    st.error("❌ Unable to fetch weather data. Please check your API key or try again later.")