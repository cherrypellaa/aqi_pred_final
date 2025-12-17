from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

DATA_PATH = "data/aqi_data.parquet"
MODEL_PATH = "models/aqi_rf_model"

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("AQI_Trainer") \
    .getOrCreate()

df = spark.read.parquet(DATA_PATH)

FEATURE_COLS = [
    "temp","humidity","pressure","wind_speed","cloudiness",
    "pm2_5","pm10","co","no2","o3"
]

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="features"
)

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="aqi",
    numTrees=100,
    maxDepth=7,
    seed=42
)

pipeline = Pipeline(stages=[assembler, rf])

model = pipeline.fit(df)

model.write().overwrite().save(MODEL_PATH)

print("✅ Model trained & saved")
