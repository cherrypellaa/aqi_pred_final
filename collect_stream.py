import os
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, explode

STREAM_DIR = "stream_coba"
OUTPUT_PATH = "data/aqi_data.parquet"

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("StreamCollector") \
    .getOrCreate()

weather_schema = StructType([
    StructField("dt", LongType()),
    StructField("main", StructType([
        StructField("temp", DoubleType()),
        StructField("humidity", DoubleType()),
        StructField("pressure", DoubleType())
    ])),
    StructField("wind", StructType([
        StructField("speed", DoubleType())
    ])),
    StructField("clouds", StructType([
        StructField("all", IntegerType())
    ]))
])

air_schema = StructType([
    StructField("list", ArrayType(StructType([
        StructField("dt", LongType()),
        StructField("main", StructType([
            StructField("aqi", IntegerType())
        ])),
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

files = [os.path.join(STREAM_DIR, f)
         for f in os.listdir(STREAM_DIR)
         if f.endswith(".json")]

if not files:
    print("No stream files")
    exit()

df_json = spark.read.schema(schema).json(files)

df = df_json.select(
    col("timestamp").alias("time_epoch"),
    col("weather.main.temp").alias("temp"),
    col("weather.main.humidity").alias("humidity"),
    col("weather.main.pressure").alias("pressure"),
    col("weather.wind.speed").alias("wind_speed"),
    col("weather.clouds.all").alias("cloudiness"),
    explode(col("air.list")).alias("air")
).select(
    "time_epoch","temp","humidity","pressure",
    "wind_speed","cloudiness",
    col("air.main.aqi").alias("aqi"),
    col("air.components.pm2_5").alias("pm2_5"),
    col("air.components.pm10").alias("pm10"),
    col("air.components.co").alias("co"),
    col("air.components.no2").alias("no2"),
    col("air.components.o3").alias("o3")
).na.drop()

df.write.mode("append").parquet(OUTPUT_PATH)

print("✅ Data appended to parquet")
