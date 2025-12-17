import os, json, time, threading, requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, explode
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

OWM_API_KEY = "bed08bec321acd6d81b2168bbfd7b2f1"
JAKARTA_COORD = {"lat": -6.2088, "lon": 106.8456}
STREAM_DIR = "stream_input"
INTERVAL_SEC = 15
os.makedirs(STREAM_DIR, exist_ok=True)

st.set_page_config(page_title="Live Weather & AQI Jakarta", layout="wide")
st.title("🌤 Live Weather & AQI Prediction - Jakarta")

def aqi_label(aqi_value):
    if aqi_value == 1:
        return "Good"
    elif aqi_value == 2:
        return "Fair"
    elif aqi_value == 3:
        return "Moderate"
    elif aqi_value == 4:
        return "Poor"
    elif aqi_value == 5:
        return "Very Poor"
    else:
        return "Unknown"

def fetch_weather():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={JAKARTA_COORD['lat']}&lon={JAKARTA_COORD['lon']}&appid={OWM_API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=8)
        return r.json() if r.status_code == 200 else {}
    except:
        return {}

def fetch_air_pollution():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={JAKARTA_COORD['lat']}&lon={JAKARTA_COORD['lon']}&appid={OWM_API_KEY}"
    try:
        r = requests.get(url, timeout=8)
        return r.json() if r.status_code == 200 else {}
    except:
        return {}

def owm_stream_writer():
    i = 0
    while True:
        weather_data = fetch_weather()
        air_data = fetch_air_pollution()
        payload = {"weather": weather_data, "air": air_data}
        filename = f"{STREAM_DIR}/owm_{i:06d}.json"
        with open(filename, "w") as f:
            json.dump(payload, f)
        i += 1
        time.sleep(INTERVAL_SEC)

threading.Thread(target=owm_stream_writer, daemon=True).start()
time.sleep(5)

spark = SparkSession.builder.master("local[*]").appName("OWM_AQI_Predict").getOrCreate()

weather_schema = StructType([
    StructField("dt", LongType(), True),
    StructField("main", StructType([
        StructField("temp", DoubleType(), True),
        StructField("humidity", DoubleType(), True),
        StructField("pressure", DoubleType(), True)
    ])),
    StructField("wind", StructType([StructField("speed", DoubleType(), True)])),
    StructField("clouds", StructType([StructField("all", IntegerType(), True)]))
])

air_schema = StructType([
    StructField("list", ArrayType(StructType([
        StructField("dt", LongType(), True),
        StructField("main", StructType([StructField("aqi", IntegerType(), True)])),
        StructField("components", StructType([
            StructField("pm2_5", DoubleType(), True),
            StructField("pm10", DoubleType(), True),
            StructField("co", DoubleType(), True),
            StructField("no2", DoubleType(), True),
            StructField("o3", DoubleType(), True)
        ]))
    ])))
])

schema = StructType([
    StructField("weather", weather_schema, True),
    StructField("air", air_schema, True)
])

time.sleep(15)
files = sorted(os.listdir(STREAM_DIR))
if not files:
    st.warning("Belum ada data stream. Tunggu sebentar...")
    st.stop()

df_json = spark.read.schema(schema).json([f"{STREAM_DIR}/{files[0]}"])
df_combined = df_json.select(
    col("weather.dt").alias("time_epoch"),
    col("weather.main.temp").alias("temp"),
    col("weather.main.humidity").alias("humidity"),
    col("weather.main.pressure").alias("pressure"),
    col("weather.wind.speed").alias("wind_speed"),
    col("weather.clouds.all").alias("cloudiness"),
    explode(col("air.list")).alias("air_event")
)

df_features = df_combined.select(
    "time_epoch","temp","humidity","pressure","wind_speed","cloudiness",
    col("air_event.main.aqi").alias("aqi"),
    col("air_event.components.pm2_5").alias("pm2_5"),
    col("air_event.components.pm10").alias("pm10"),
    col("air_event.components.co").alias("co"),
    col("air_event.components.no2").alias("no2"),
    col("air_event.components.o3").alias("o3")
).na.drop()

assembler = VectorAssembler(
    inputCols=["temp","humidity","pressure","wind_speed","cloudiness","pm2_5","pm10","co","no2","o3"],
    outputCol="features"
)

rf = RandomForestRegressor(featuresCol="features", labelCol="aqi")
pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(df_features)

trend_history = []
chart_placeholder = st.empty()
metric_placeholder = st.empty()

while True:
    files = sorted(os.listdir(STREAM_DIR))
    if not files:
        time.sleep(5)
        continue

    latest_file = files[-1]
    df_json = spark.read.schema(schema).json(f"{STREAM_DIR}/{latest_file}")
    df_combined = df_json.select(
        col("weather.dt").alias("time_epoch"),
        col("weather.main.temp").alias("temp"),
        col("weather.main.humidity").alias("humidity"),
        col("weather.main.pressure").alias("pressure"),
        col("weather.wind.speed").alias("wind_speed"),
        col("weather.clouds.all").alias("cloudiness"),
        explode(col("air.list")).alias("air_event")
    )

    df_features = df_combined.select(
        "time_epoch","temp","humidity","pressure","wind_speed","cloudiness",
        col("air_event.main.aqi").alias("aqi"),
        col("air_event.components.pm2_5").alias("pm2_5"),
        col("air_event.components.pm10").alias("pm10"),
        col("air_event.components.co").alias("co"),
        col("air_event.components.no2").alias("no2"),
        col("air_event.components.o3").alias("o3")
    ).na.drop()

    pdf = model.transform(df_features).select(
        "time_epoch","temp","humidity","pressure","wind_speed","cloudiness",
        "aqi","pm2_5","pm10","co","no2","o3","prediction"
    ).toPandas()

    pdf["prediction_int"] = pdf["prediction"].round().astype(int)
    trend_history.append(pdf.iloc[0].to_dict())
    trend_df = pd.DataFrame(trend_history)
    trend_df["time"] = pd.to_datetime(trend_df["time_epoch"], unit='s')

    if len(trend_df) < 2:
        time.sleep(INTERVAL_SEC)
        continue  

    latest = trend_df.iloc[-1]

    with metric_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡 Temperature (°C)", f"{latest.temp:.1f}")
        col2.metric("💧 Humidity (%)", f"{latest.humidity:.1f}")
        col3.metric("💨 Wind Speed (m/s)", f"{latest.wind_speed:.1f}")
        col4.metric("☁ Cloudiness (%)", f"{latest.cloudiness:.0f}")

        col5, col6, col7, col8, col9 = st.columns(5)
        col5.metric(
        "🔴 AQI",
        f"{latest.aqi} ({aqi_label(latest.aqi)})",
        delta=f"{latest.prediction - latest.aqi:+.1f}"  
        )

        st.markdown(f"**Predicted AQI:** {latest.prediction_int} ({aqi_label(latest.prediction_int)})")

        col6.metric("PM2.5 (µg/m³)", f"{latest.pm2_5:.1f}")
        col7.metric("PM10 (µg/m³)", f"{latest.pm10:.1f}")
        col8.metric("CO (µg/m³)", f"{latest.co:.2f}")
        col9.metric("NO2 (µg/m³)", f"{latest.no2:.2f}")

    fig, ax = plt.subplots(2,1, figsize=(12,6))
    ax[0].plot(trend_df["time"], trend_df["temp"], label="Temp (°C)", marker='o')
    ax[0].plot(trend_df["time"], trend_df["humidity"], label="Humidity (%)", marker='o')
    ax[0].plot(trend_df["time"], trend_df["wind_speed"], label="Wind Speed (m/s)", marker='o')
    ax[0].legend()
    ax[0].set_title("Weather Trend")

    ax[1].plot(trend_df["time"], trend_df["aqi"], label="AQI", color="red", marker='o')
    ax[1].plot(trend_df["time"], trend_df["prediction_int"], label="Predicted AQI", linestyle="--", color="orange", marker='o')
    ax[1].legend()
    ax[1].set_title("Air Quality Trend")

    plt.tight_layout()
    chart_placeholder.pyplot(fig)
    plt.close(fig)

    time.sleep(INTERVAL_SEC)
