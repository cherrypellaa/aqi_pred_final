import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from streamlit_autorefresh import st_autorefresh
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, explode
from pyspark.ml import PipelineModel

STREAM_DIR = "stream_coba"
MODEL_PATH = "models/aqi_rf_model"
REFRESH_SEC = 10
FORECAST_MINUTES = 10

st.set_page_config(page_title="Live Weather & AQI Jakarta", layout="wide")
st.title("🌤 Live Weather & AQI Jakarta")
st.caption("Spark Streaming Consumer + Offline Trained ML Model")
st_autorefresh(interval=REFRESH_SEC * 1000, key="refresh")

if "trend_history" not in st.session_state:
    st.session_state.trend_history = []

if "last_file" not in st.session_state:
    st.session_state.last_file = None

@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .master("local[*]") \
        .appName("AQI_Streamlit") \
        .getOrCreate()

@st.cache_resource
def load_model():
    return PipelineModel.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

spark = get_spark()
model = load_model()

weather_schema = StructType([
    StructField("dt", LongType()),
    StructField("main", StructType([
        StructField("temp", DoubleType()),
        StructField("humidity", DoubleType()),
        StructField("pressure", DoubleType())
    ])),
    StructField("wind", StructType([StructField("speed", DoubleType())])),
    StructField("clouds", StructType([StructField("all", IntegerType())]))
])

air_schema = StructType([
    StructField("list", ArrayType(StructType([
        StructField("dt", LongType()),
        StructField("main", StructType([StructField("aqi", IntegerType())])),
        StructField("components", StructType([
            StructField("pm2_5", DoubleType()),
            StructField("pm10", DoubleType()),
            StructField("co", DoubleType()),
            StructField("no2", DoubleType()),
            StructField("o3", DoubleType())
        ]))
    ])))
])

schema = StructType([
    StructField("timestamp", LongType()),
    StructField("weather", weather_schema),
    StructField("air", air_schema)
])

def aqi_label(v):
    try:
        v = int(round(v))
    except:
        return "Unknown"
    return {1:"Good",2:"Fair",3:"Moderate",4:"Poor",5:"Very Poor"}.get(v,"Unknown")

files = [os.path.join(STREAM_DIR,f) for f in os.listdir(STREAM_DIR) if f.endswith(".json")]
if not files:
    st.warning("⏳ Menunggu data stream...")
    st.stop()

latest_file = max(files, key=os.path.getmtime)
if latest_file == st.session_state.last_file:
    st.stop()

st.session_state.last_file = latest_file
st.caption(f"📄 File: `{os.path.basename(latest_file)}`")

df = spark.read.schema(schema).json(latest_file) \
    .select(
        col("timestamp").alias("time_epoch"),
        col("weather.main.temp").alias("temp"),
        col("weather.main.humidity").alias("humidity"),
        col("weather.main.pressure").alias("pressure"),
        col("weather.wind.speed").alias("wind_speed"),
        col("weather.clouds.all").alias("cloudiness"),
        explode(col("air.list")).alias("air")
    ).select(
        "time_epoch","temp","humidity","pressure","wind_speed","cloudiness",
        col("air.main.aqi").alias("aqi"),
        col("air.components.pm2_5").alias("pm2_5"),
        col("air.components.pm10").alias("pm10"),
        col("air.components.co").alias("co"),
        col("air.components.no2").alias("no2"),
        col("air.components.o3").alias("o3")
    ).na.drop()

pdf = df.toPandas().iloc[0]

FEATURE_COLS = [
    "time_epoch","temp","humidity","pressure",
    "wind_speed","cloudiness",
    "pm2_5","pm10","co","no2","o3"
]

base_row = {k: float(pdf[k]) for k in FEATURE_COLS}

pred_aqi = forecast_aqi = None
if model:
    spark_now = spark.createDataFrame([base_row])
    pred_aqi = model.transform(spark_now).select("prediction").toPandas().iloc[0,0]

    future_row = base_row.copy()
    future_row["time_epoch"] += FORECAST_MINUTES * 60
    spark_future = spark.createDataFrame([future_row])
    forecast_aqi = model.transform(spark_future).select("prediction").toPandas().iloc[0,0]

history = base_row.copy()
history.update({
    "aqi": int(pdf["aqi"]),
    "pred_aqi": pred_aqi,
    "forecast_10min": forecast_aqi
})

st.session_state.trend_history.append(history)
trend_df = pd.DataFrame(st.session_state.trend_history)
trend_df["time"] = pd.to_datetime(trend_df["time_epoch"], unit="s")

latest = trend_df.iloc[-1]

st.subheader("📌 Air Quality Overview")

c1,c2,c3 = st.columns(3)
c1.metric("🔮 Predicted AQI (Now)", f"{int(round(pred_aqi))} ({aqi_label(pred_aqi)})")
c2.metric("⏩ Forecast AQI (+10 min)", f"{int(round(forecast_aqi))} ({aqi_label(forecast_aqi)})")
c3.metric("🔴 Current AQI", f"{int(pdf.aqi)} ({aqi_label(pdf.aqi)})")

st.subheader("🌦 Weather Conditions")
w1,w2,w3,w4 = st.columns(4)
w1.metric("🌡 Temp (°C)", f"{latest.temp:.1f}")
w2.metric("💧 Humidity (%)", f"{latest.humidity:.0f}")
w3.metric("💨 Wind (m/s)", f"{latest.wind_speed:.1f}")
w4.metric("☁ Cloud (%)", f"{latest.cloudiness:.0f}")

st.subheader("🧪 Air Pollutants")
p1,p2,p3,p4,p5 = st.columns(5)
p1.metric("PM2.5", f"{latest.pm2_5:.1f}")
p2.metric("PM10", f"{latest.pm10:.1f}")
p3.metric("CO", f"{latest.co:.2f}")
p4.metric("NO₂", f"{latest.no2:.2f}")
p5.metric("O₃", f"{latest.o3:.2f}")

st.subheader("📊 Recent Data")
st.dataframe(trend_df.tail(10), use_container_width=True)

fig, ax = plt.subplots(2,1, figsize=(12,7))

ax[0].plot(trend_df["time"], trend_df["temp"], label="Temp")
ax[0].plot(trend_df["time"], trend_df["humidity"], label="Humidity")
ax[0].legend()
ax[0].set_title("Weather Trend")

ax[1].plot(trend_df["time"], trend_df["aqi"], label="Actual AQI", marker="o")
ax[1].plot(trend_df["time"], trend_df["pred_aqi"], label="Predicted AQI", linestyle="--")

future_time = trend_df["time"] + pd.Timedelta(minutes=FORECAST_MINUTES)
ax[1].plot(future_time, trend_df["forecast_10min"], label="Forecast AQI (+10 min)", linestyle=":", marker="x")

ax[1].axvline(trend_df["time"].iloc[-1], linestyle="--", color="gray")
ax[1].legend()
ax[1].set_title("Air Quality Trend")

st.pyplot(fig)
