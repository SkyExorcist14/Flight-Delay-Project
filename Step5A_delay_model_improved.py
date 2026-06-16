import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from py4j.protocol import Py4JJavaError

# ===============================================================
# ENVIRONMENT SETUP
# ===============================================================
PYTHON_EXE = r"C:\Python311\python.exe"
JAVA_HOME = os.path.abspath(r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot")
HADOOP_HOME_WIN = os.path.abspath(r"D:\hadoop")
HADOOP_HOME_JAVA = HADOOP_HOME_WIN.replace("\\", "/")
HADOOP_BIN_WIN = os.path.join(HADOOP_HOME_WIN, "bin")
HADOOP_BIN_JAVA = HADOOP_BIN_WIN.replace("\\", "/")


def prepend_path(path_value: str) -> None:
    current = os.environ.get("PATH", "")
    parts = current.split(";") if current else []
    if path_value and path_value not in parts:
        os.environ["PATH"] = path_value + (";" + current if current else "")


os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["HADOOP_HOME"] = HADOOP_HOME_JAVA
os.environ["hadoop.home.dir"] = HADOOP_HOME_JAVA
os.environ["PYSPARK_PYTHON"] = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE

prepend_path(os.path.join(JAVA_HOME, "bin"))
prepend_path(HADOOP_BIN_WIN)

if hasattr(os, "add_dll_directory"):
    for dll_dir in [os.path.join(JAVA_HOME, "bin"), HADOOP_BIN_WIN]:
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)

if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

if not os.path.exists(PYTHON_EXE):
    print(f"ERROR: Python not found at {PYTHON_EXE}")
    sys.exit(1)

if not os.path.exists(os.path.join(HADOOP_BIN_WIN, "winutils.exe")):
    print(f"ERROR: winutils.exe not found in {HADOOP_BIN_WIN}")
    sys.exit(1)

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    dayofweek,
    floor,
    month,
    round as spark_round,
    when,
)
from pyspark.sql.types import DoubleType

# ===============================================================
# PATHS / SETTINGS
# ===============================================================
PROJECT_ROOT = r"D:\UMBC All Data\DATA 603\flight_project"
delay_folder = rf"{PROJECT_ROOT}\Data\Delays"
processed = rf"{PROJECT_ROOT}\Data\Processed"
model_dir = rf"{PROJECT_ROOT}\Data\Models"

MODEL_OUTPUT_DIR = os.path.join(model_dir, "delay_rf_model_v2")
METRICS_OUTPUT = os.path.join(processed, "delay_model_metrics_v2.json")
IMPORTANCE_OUTPUT = os.path.join(processed, "delay_feature_importance_v2.csv")
COMPARISON_OUTPUT = os.path.join(processed, "delay_model_comparison_v2.json")

SAMPLE_FRACTION = 0.20
SEED = 42

os.makedirs(model_dir, exist_ok=True)
os.makedirs(processed, exist_ok=True)


def to_spark_uri(path_str: str) -> str:
    return Path(path_str).resolve().as_uri()


def reset_output_path(path_str: str) -> None:
    if os.path.isdir(path_str):
        shutil.rmtree(path_str)
    elif os.path.isfile(path_str):
        os.remove(path_str)


def save_spark_model(model, output_path: str, label: str) -> None:
    reset_output_path(output_path)
    output_uri = to_spark_uri(output_path)

    try:
        model.write().overwrite().save(output_uri)
        print(f"OK: {label} saved to {output_path}")
    except Py4JJavaError:
        print(f"ERROR: Failed to save {label} to {output_path}")
        raise


def load_json_if_exists(path_str: str):
    if os.path.exists(path_str):
        with open(path_str, "r") as file_obj:
            return json.load(file_obj)
    return None


# ===============================================================
# START SPARK
# ===============================================================
print("Starting Spark...")
print(f"Hadoop home: {HADOOP_HOME_JAVA}")
print(f"Hadoop bin : {HADOOP_BIN_WIN}")

spark = (
    SparkSession.builder
    .appName("ImprovedDelayModelOnly")
    .config("spark.driver.memory", "8g")
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.pyspark.python", PYTHON_EXE)
    .config("spark.pyspark.driver.python", PYTHON_EXE)
    .config(
        "spark.driver.extraJavaOptions",
        f"-Djava.library.path={HADOOP_BIN_JAVA} -Dhadoop.home.dir={HADOOP_HOME_JAVA}",
    )
    .config(
        "spark.executor.extraJavaOptions",
        f"-Djava.library.path={HADOOP_BIN_JAVA} -Dhadoop.home.dir={HADOOP_HOME_JAVA}",
    )
    .config("spark.hadoop.hadoop.home.dir", HADOOP_HOME_JAVA)
    .config("spark.hadoop.home.dir", HADOOP_HOME_JAVA)
    .config("spark.hadoop.fs.file.impl.disable.cache", "true")
    .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

hconf = spark.sparkContext._jsc.hadoopConfiguration()
hconf.set("hadoop.home.dir", HADOOP_HOME_JAVA)
hconf.set("HADOOP_HOME", HADOOP_HOME_JAVA)
hconf.set("fs.file.impl.disable.cache", "true")
hconf.set("mapreduce.fileoutputcommitter.algorithm.version", "2")

print("OK: Spark started\n")

# ===============================================================
# LOAD AND CLEAN DELAY DATA
# ===============================================================
print("=" * 60)
print("IMPROVED DELAY MODEL - TRAINING ONLY")
print("=" * 60)

delay_files = sorted(Path(delay_folder).glob("*.csv"))
delay_paths = [path.resolve().as_posix() for path in delay_files]
print(f"Loading {len(delay_paths)} delay files...")

raw_delays = spark.read.csv(delay_paths, header=True, inferSchema=True)

if "Unnamed: 27" in raw_delays.columns:
    raw_delays = raw_delays.drop("Unnamed: 27")

delay_base = (
    raw_delays.filter(
        col("DEP_DELAY").isNotNull()
        & col("ORIGIN").isNotNull()
        & col("DEST").isNotNull()
        & col("OP_CARRIER").isNotNull()
        & col("DISTANCE").isNotNull()
        & col("CRS_DEP_TIME").isNotNull()
        & col("CRS_ELAPSED_TIME").isNotNull()
    )
    .withColumn("DEP_DELAY", col("DEP_DELAY").cast(DoubleType()))
    .withColumn("DISTANCE", col("DISTANCE").cast(DoubleType()))
    .withColumn("CRS_DEP_TIME", col("CRS_DEP_TIME").cast(DoubleType()))
    .withColumn("CRS_ELAPSED_TIME", col("CRS_ELAPSED_TIME").cast(DoubleType()))
    .withColumn("WEATHER_DELAY", col("WEATHER_DELAY").cast(DoubleType()))
    .filter(
        (col("DEP_DELAY") >= -30)
        & (col("DEP_DELAY") <= 240)
        & (col("DISTANCE") > 0)
        & (col("CRS_ELAPSED_TIME") > 0)
    )
    .withColumn("month", month(col("FL_DATE")))
    .withColumn("day_of_week", dayofweek(col("FL_DATE")))
    .withColumn("dep_hour", floor(col("CRS_DEP_TIME") / 100.0).cast(DoubleType()))
    .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1.0).otherwise(0.0))
    .withColumn("is_summer", when(col("month").isin([6, 7, 8]), 1.0).otherwise(0.0))
    .withColumn("is_holiday_season", when(col("month").isin([11, 12]), 1.0).otherwise(0.0))
    .withColumn("is_winter", when(col("month").isin([12, 1, 2]), 1.0).otherwise(0.0))
    .withColumn(
        "weather_delay_signal",
        when(col("WEATHER_DELAY").isNull(), 0.0).otherwise(col("WEATHER_DELAY")),
    )
    .select(
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        "month",
        "day_of_week",
        "dep_hour",
        "is_weekend",
        "is_summer",
        "is_holiday_season",
        "is_winter",
        "DISTANCE",
        "CRS_ELAPSED_TIME",
        "weather_delay_signal",
        "DEP_DELAY",
    )
)

delay_sample = delay_base.sample(fraction=SAMPLE_FRACTION, seed=SEED)
sample_count = delay_sample.count()
print(f"Training sample: {sample_count:,} records ({int(SAMPLE_FRACTION * 100)}% of data)")

train_delay, test_delay = delay_sample.randomSplit([0.8, 0.2], seed=SEED)
train_count = train_delay.count()
test_count = test_delay.count()

print(f"Training rows: {train_count:,}")
print(f"Testing rows : {test_count:,}")

global_avg_delay = train_delay.agg(avg("DEP_DELAY").alias("global_avg_delay")).collect()[0]["global_avg_delay"]

origin_stats = train_delay.groupBy("ORIGIN").agg(
    avg("DEP_DELAY").alias("origin_avg_delay"),
)

dest_stats = train_delay.groupBy("DEST").agg(
    avg("DEP_DELAY").alias("dest_avg_delay"),
)

carrier_stats = train_delay.groupBy("OP_CARRIER").agg(
    avg("DEP_DELAY").alias("carrier_avg_delay"),
)

route_stats = train_delay.groupBy("ORIGIN", "DEST").agg(
    avg("DEP_DELAY").alias("route_avg_delay"),
)

carrier_route_stats = train_delay.groupBy("OP_CARRIER", "ORIGIN", "DEST").agg(
    avg("DEP_DELAY").alias("carrier_route_avg_delay"),
)

dep_hour_stats = train_delay.groupBy("dep_hour").agg(
    avg("DEP_DELAY").alias("dep_hour_avg_delay"),
)

origin_weather_stats = train_delay.groupBy("ORIGIN").agg(
    avg("weather_delay_signal").alias("origin_weather_delay_avg"),
)


def enrich_delay_frame(frame):
    enriched = (
        frame.join(origin_stats, on="ORIGIN", how="left")
        .join(dest_stats, on="DEST", how="left")
        .join(carrier_stats, on="OP_CARRIER", how="left")
        .join(route_stats, on=["ORIGIN", "DEST"], how="left")
        .join(carrier_route_stats, on=["OP_CARRIER", "ORIGIN", "DEST"], how="left")
        .join(dep_hour_stats, on="dep_hour", how="left")
        .join(origin_weather_stats, on="ORIGIN", how="left")
        .fillna(
            {
                "origin_avg_delay": float(global_avg_delay),
                "dest_avg_delay": float(global_avg_delay),
                "carrier_avg_delay": float(global_avg_delay),
                "route_avg_delay": float(global_avg_delay),
                "carrier_route_avg_delay": float(global_avg_delay),
                "dep_hour_avg_delay": float(global_avg_delay),
                "origin_weather_delay_avg": 0.0,
            }
        )
    )
    return enriched


train_enriched = enrich_delay_frame(train_delay)
test_enriched = enrich_delay_frame(test_delay)

num_carriers = train_enriched.select("OP_CARRIER").distinct().count()
max_bins = 64

print(f"Unique carriers : {num_carriers}")
print(f"maxBins         : {max_bins}")
print("Training improved Gradient Boosted Trees model... this may take a while.")

carrier_idx = StringIndexer(inputCol="OP_CARRIER", outputCol="carrier_idx", handleInvalid="keep")

feature_columns = [
    "carrier_idx",
    "month",
    "day_of_week",
    "dep_hour",
    "is_weekend",
    "is_summer",
    "is_holiday_season",
    "is_winter",
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "weather_delay_signal",
    "origin_avg_delay",
    "dest_avg_delay",
    "carrier_avg_delay",
    "route_avg_delay",
    "carrier_route_avg_delay",
    "dep_hour_avg_delay",
    "origin_weather_delay_avg",
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
    handleInvalid="keep",
)

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="DEP_DELAY",
    maxIter=30,
    maxDepth=6,
    maxBins=max_bins,
    stepSize=0.08,
    subsamplingRate=0.8,
    seed=SEED,
)

pipeline = Pipeline(stages=[carrier_idx, assembler, gbt])
delay_model_v2 = pipeline.fit(train_enriched)
print("OK: Improved delay model trained")

preds = delay_model_v2.transform(test_enriched)

rmse = RegressionEvaluator(
    labelCol="DEP_DELAY",
    predictionCol="prediction",
    metricName="rmse",
).evaluate(preds)
mae = RegressionEvaluator(
    labelCol="DEP_DELAY",
    predictionCol="prediction",
    metricName="mae",
).evaluate(preds)
r2 = RegressionEvaluator(
    labelCol="DEP_DELAY",
    predictionCol="prediction",
    metricName="r2",
).evaluate(preds)

print("\nIMPROVED DELAY MODEL PERFORMANCE")
print(f"RMSE: {rmse:.2f} minutes")
print(f"MAE : {mae:.2f} minutes")
print(f"R2  : {r2:.4f}")

gbt_stage = delay_model_v2.stages[-1]
importances = gbt_stage.featureImportances.toArray()
importance_pdf = pd.DataFrame(
    {"feature": feature_columns, "importance": importances}
).sort_values("importance", ascending=False)

print("\nTOP FEATURE IMPORTANCE")
print(importance_pdf.head(15).to_string(index=False))
importance_pdf.to_csv(IMPORTANCE_OUTPUT, index=False)
print(f"OK: Saved {os.path.basename(IMPORTANCE_OUTPUT)}")

save_spark_model(delay_model_v2, MODEL_OUTPUT_DIR, "improved delay GBT model")

metrics_v2 = {
    "model": "GBTRegressor",
    "target": "DEP_DELAY",
    "sample_fraction": SAMPLE_FRACTION,
    "rmse": round(rmse, 2),
    "mae": round(mae, 2),
    "r2": round(r2, 4),
    "max_iter": 30,
    "max_depth": 6,
    "max_bins": max_bins,
    "feature_count": len(feature_columns),
    "model_path": MODEL_OUTPUT_DIR,
}

with open(METRICS_OUTPUT, "w") as file_obj:
    json.dump(metrics_v2, file_obj, indent=2)
print(f"OK: Saved {os.path.basename(METRICS_OUTPUT)}")

baseline_metrics = load_json_if_exists(os.path.join(processed, "delay_model_metrics.json"))
comparison = {"baseline": baseline_metrics, "improved_v2": metrics_v2}

if baseline_metrics:
    comparison["delta"] = {
        "rmse_change": round(metrics_v2["rmse"] - baseline_metrics["rmse"], 2),
        "mae_change": round(metrics_v2["mae"] - baseline_metrics["mae"], 2),
        "r2_change": round(metrics_v2["r2"] - baseline_metrics["r2"], 4),
    }

with open(COMPARISON_OUTPUT, "w") as file_obj:
    json.dump(comparison, file_obj, indent=2)
print(f"OK: Saved {os.path.basename(COMPARISON_OUTPUT)}")

spark.stop()

print("\nDone.")
print("This run only retrained the delay model.")
print("Baseline Step5_ML_Model.py remains untouched.")
print("New outputs:")
print(f"  Model : {MODEL_OUTPUT_DIR}")
print(f"  Metrics: {METRICS_OUTPUT}")
print(f"  Compare: {COMPARISON_OUTPUT}")
