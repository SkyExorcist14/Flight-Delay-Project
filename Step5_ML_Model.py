import glob
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
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc, round as spark_round, when
from pyspark.sql.functions import dayofweek, month, isnan
from pyspark.sql.types import DoubleType

# ===============================================================
# PATHS
# ===============================================================
PROJECT_ROOT = r"D:\UMBC All Data\DATA 603\flight_project"
delay_folder = rf"{PROJECT_ROOT}\Data\Delays"
fare_folder = rf"{PROJECT_ROOT}\Data\Fare"
processed = rf"{PROJECT_ROOT}\Data\Processed"
model_dir = rf"{PROJECT_ROOT}\Data\Models"
DELAY_SAMPLE_FRACTION = 0.40
FARE_SAMPLE_FRACTION = 0.40

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
    except Py4JJavaError as exc:
        error_text = str(exc)
        print(f"ERROR: Failed to save {label}")
        print(f"Target path: {output_path}")

        if "NativeIO$Windows.access0" in error_text:
            print("Hadoop native Windows library was not loaded by the JVM.")
            print("Step 5 now adds D:\\hadoop\\bin to PATH and java.library.path.")
        elif "ChangeFileModeByMask" in error_text:
            print("winutils.exe hit a Windows permission problem while saving.")
            print("Run the script from a folder where your user has full write access.")

        raise


def save_pandas_csv(df, output_name: str) -> None:
    df.toPandas().to_csv(os.path.join(processed, output_name), index=False)
    print(f"OK: Saved {output_name}")


# ===============================================================
# START SPARK
# ===============================================================
print("Starting Spark...")
print(f"Hadoop home: {HADOOP_HOME_JAVA}")
print(f"Hadoop bin : {HADOOP_BIN_WIN}")

spark = (
    SparkSession.builder
    .appName("FlightMLModels")
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
# MODEL 1 - DELAY PREDICTOR
# ===============================================================
print("=" * 60)
print("MODEL 1: DELAY PREDICTOR (Random Forest Regressor)")
print("=" * 60)

delay_files = glob.glob(os.path.join(delay_folder, "*.csv"))
delay_files_spark = [Path(path).resolve().as_posix() for path in delay_files]

print(f"Loading {len(delay_files_spark)} delay files...")

raw_delays = spark.read.csv(delay_files_spark, header=True, inferSchema=True)

if "Unnamed: 27" in raw_delays.columns:
    raw_delays = raw_delays.drop("Unnamed: 27")

delay_data = (
    raw_delays.filter(
        col("DEP_DELAY").isNotNull()
        & col("ORIGIN").isNotNull()
        & col("DEST").isNotNull()
        & col("OP_CARRIER").isNotNull()
        & col("DISTANCE").isNotNull()
    )
    .withColumn("month", month(col("FL_DATE")))
    .withColumn("day_of_week", dayofweek(col("FL_DATE")))
    .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1.0).otherwise(0.0))
    .withColumn("is_summer", when(col("month").isin([6, 7, 8]), 1.0).otherwise(0.0))
    .withColumn("is_holiday_season", when(col("month").isin([11, 12]), 1.0).otherwise(0.0))
    .withColumn("is_winter", when(col("month").isin([12, 1, 2]), 1.0).otherwise(0.0))
    .withColumn("DEP_DELAY", col("DEP_DELAY").cast(DoubleType()))
    .withColumn("DISTANCE", col("DISTANCE").cast(DoubleType()))
    .select(
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        "month",
        "day_of_week",
        "is_weekend",
        "is_summer",
        "is_holiday_season",
        "is_winter",
        "DISTANCE",
        "DEP_DELAY",
    )
)

delay_sample = delay_data.sample(fraction=DELAY_SAMPLE_FRACTION, seed=42)
total_sample = delay_sample.count()
print(f"Training sample: {total_sample:,} records ({int(DELAY_SAMPLE_FRACTION * 100)}% of data)")

num_origins = delay_sample.select("ORIGIN").distinct().count()
num_dests = delay_sample.select("DEST").distinct().count()
num_carriers = delay_sample.select("OP_CARRIER").distinct().count()

max_bins = max(num_origins, num_dests, num_carriers) + 50
max_bins = ((max_bins // 128) + 1) * 128

print(f"Unique origins  : {num_origins}")
print(f"Unique dests    : {num_dests}")
print(f"Unique carriers : {num_carriers}")
print(f"maxBins         : {max_bins}")

carrier_idx = StringIndexer(
    inputCol="OP_CARRIER",
    outputCol="carrier_idx",
    handleInvalid="keep",
)
origin_idx = StringIndexer(
    inputCol="ORIGIN",
    outputCol="origin_idx",
    handleInvalid="keep",
)
dest_idx = StringIndexer(
    inputCol="DEST",
    outputCol="dest_idx",
    handleInvalid="keep",
)

assembler_delay = VectorAssembler(
    inputCols=[
        "carrier_idx",
        "origin_idx",
        "dest_idx",
        "month",
        "day_of_week",
        "is_weekend",
        "is_summer",
        "is_holiday_season",
        "is_winter",
        "DISTANCE",
    ],
    outputCol="features",
    handleInvalid="keep",
)

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="DEP_DELAY",
    numTrees=50,
    maxDepth=8,
    maxBins=max_bins,
    minInstancesPerNode=10,
    featureSubsetStrategy="auto",
    seed=42,
)

delay_pipeline = Pipeline(stages=[carrier_idx, origin_idx, dest_idx, assembler_delay, rf])

train_delay, test_delay = delay_sample.randomSplit([0.8, 0.2], seed=42)

print(f"Training rows: {train_delay.count():,}")
print(f"Testing rows : {test_delay.count():,}")
print("Training Random Forest... this may take a while.")

delay_trained = delay_pipeline.fit(train_delay)
print("OK: Delay model trained")

delay_preds = delay_trained.transform(test_delay)

rmse_delay = RegressionEvaluator(
    labelCol="DEP_DELAY",
    predictionCol="prediction",
    metricName="rmse",
).evaluate(delay_preds)
r2_delay = RegressionEvaluator(
    labelCol="DEP_DELAY",
    predictionCol="prediction",
    metricName="r2",
).evaluate(delay_preds)
mae_delay = RegressionEvaluator(
    labelCol="DEP_DELAY",
    predictionCol="prediction",
    metricName="mae",
).evaluate(delay_preds)

print("\nDELAY MODEL PERFORMANCE")
print(f"RMSE: {rmse_delay:.2f} minutes")
print(f"MAE : {mae_delay:.2f} minutes")
print(f"R2  : {r2_delay:.4f}")

rf_stage = delay_trained.stages[-1]
importances = rf_stage.featureImportances.toArray()
feature_names = [
    "carrier",
    "origin",
    "destination",
    "month",
    "day_of_week",
    "is_weekend",
    "is_summer",
    "is_holiday_season",
    "is_winter",
    "distance",
]

imp_pdf = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values("importance", ascending=False)

print("\nDELAY FEATURE IMPORTANCE")
print(imp_pdf.to_string(index=False))
imp_pdf.to_csv(os.path.join(processed, "delay_feature_importance.csv"), index=False)
print("OK: Saved delay_feature_importance.csv")

delay_model_path = os.path.join(model_dir, "delay_rf_model")
save_spark_model(delay_trained, delay_model_path, "delay Random Forest model")

delay_metrics = {
    "model": "RandomForestRegressor",
    "target": "DEP_DELAY",
    "rmse": round(rmse_delay, 2),
    "mae": round(mae_delay, 2),
    "r2": round(r2_delay, 4),
    "num_trees": 50,
    "max_depth": 8,
    "max_bins": max_bins,
}
with open(os.path.join(processed, "delay_model_metrics.json"), "w") as file_obj:
    json.dump(delay_metrics, file_obj, indent=2)
print("OK: Saved delay_model_metrics.json\n")

# ===============================================================
# MODEL 2 - PRICE PREDICTOR
# ===============================================================
print("=" * 60)
print("MODEL 2: PRICE PREDICTOR (Gradient Boosted Trees)")
print("=" * 60)

fare_files = glob.glob(os.path.join(fare_folder, "*.csv"))
fare_files_spark = [Path(path).resolve().as_posix() for path in fare_files]

print(f"Loading {len(fare_files_spark)} fare files...")

raw_fares = spark.read.csv(fare_files_spark, header=True, inferSchema=True)

fare_data = (
    raw_fares.filter(
        col("ITIN_FARE").isNotNull()
        & col("ORIGIN").isNotNull()
        & col("DISTANCE").isNotNull()
        & col("YEAR").isNotNull()
        & col("QUARTER").isNotNull()
        & (col("ITIN_FARE") > 10)
        & (col("ITIN_FARE") < 5000)
        & (col("DISTANCE") > 0)
    )
    .withColumn("ITIN_FARE", col("ITIN_FARE").cast(DoubleType()))
    .withColumn("DISTANCE", col("DISTANCE").cast(DoubleType()))
    .withColumn("ITIN_YIELD", col("ITIN_YIELD").cast(DoubleType()))
    .withColumn("QUARTER", col("QUARTER").cast(DoubleType()))
    .withColumn("YEAR", col("YEAR").cast(DoubleType()))
    .withColumn(
        "yield_per_mile",
        when(
            col("ITIN_YIELD").isNotNull()
            & (~isnan(col("ITIN_YIELD")))
            & (col("ITIN_YIELD") > 0),
            col("ITIN_YIELD"),
        ).otherwise(col("ITIN_FARE") / col("DISTANCE"))
    )
    .withColumn("is_peak_quarter", when(col("QUARTER").isin([2.0, 3.0]), 1.0).otherwise(0.0))
    .filter(
        col("ITIN_FARE").isNotNull()
        & (~isnan(col("ITIN_FARE")))
        & col("DISTANCE").isNotNull()
        & (~isnan(col("DISTANCE")))
        & col("QUARTER").isNotNull()
        & (~isnan(col("QUARTER")))
        & col("YEAR").isNotNull()
        & (~isnan(col("YEAR")))
        & col("yield_per_mile").isNotNull()
        & (~isnan(col("yield_per_mile")))
        & (col("yield_per_mile") > 0)
        & (col("yield_per_mile") < 1000)
    )
    .select(
        "ORIGIN",
        "QUARTER",
        "YEAR",
        "DISTANCE",
        "yield_per_mile",
        "is_peak_quarter",
        "ITIN_FARE",
    )
)

fare_sample = fare_data.sample(fraction=FARE_SAMPLE_FRACTION, seed=42)
total_fare_sample = fare_sample.count()
print(f"Training sample: {total_fare_sample:,} records ({int(FARE_SAMPLE_FRACTION * 100)}% of data)")

num_fare_origins = fare_sample.select("ORIGIN").distinct().count()
fare_max_bins = ((num_fare_origins // 128) + 1) * 128

print(f"Unique fare origins: {num_fare_origins}")
print(f"maxBins            : {fare_max_bins}")

origin_idx_fare = StringIndexer(
    inputCol="ORIGIN",
    outputCol="origin_idx",
    handleInvalid="keep",
)

assembler_fare = VectorAssembler(
    inputCols=[
        "origin_idx",
        "QUARTER",
        "YEAR",
        "DISTANCE",
        "yield_per_mile",
        "is_peak_quarter",
    ],
    outputCol="features",
    handleInvalid="keep",
)

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="ITIN_FARE",
    maxIter=50,
    maxDepth=6,
    maxBins=fare_max_bins,
    stepSize=0.1,
    seed=42,
)

fare_pipeline = Pipeline(stages=[origin_idx_fare, assembler_fare, gbt])

train_fare, test_fare = fare_sample.randomSplit([0.8, 0.2], seed=42)

print(f"Training rows: {train_fare.count():,}")
print(f"Testing rows : {test_fare.count():,}")
print("Training GBT price model... this may take a while.")

fare_trained = fare_pipeline.fit(train_fare)
print("OK: Price model trained")

fare_preds = fare_trained.transform(test_fare)

rmse_fare = RegressionEvaluator(
    labelCol="ITIN_FARE",
    predictionCol="prediction",
    metricName="rmse",
).evaluate(fare_preds)
r2_fare = RegressionEvaluator(
    labelCol="ITIN_FARE",
    predictionCol="prediction",
    metricName="r2",
).evaluate(fare_preds)
mae_fare = RegressionEvaluator(
    labelCol="ITIN_FARE",
    predictionCol="prediction",
    metricName="mae",
).evaluate(fare_preds)

print("\nPRICE MODEL PERFORMANCE")
print(f"RMSE: ${rmse_fare:.2f}")
print(f"MAE : ${mae_fare:.2f}")
print(f"R2  : {r2_fare:.4f}")

fare_model_path = os.path.join(model_dir, "fare_gbt_model")
save_spark_model(fare_trained, fare_model_path, "fare GBT model")

fare_metrics = {
    "model": "GBTRegressor",
    "target": "ITIN_FARE",
    "rmse": round(rmse_fare, 2),
    "mae": round(mae_fare, 2),
    "r2": round(r2_fare, 4),
    "max_iter": 50,
    "max_depth": 6,
    "max_bins": fare_max_bins,
}
with open(os.path.join(processed, "fare_model_metrics.json"), "w") as file_obj:
    json.dump(fare_metrics, file_obj, indent=2)
print("OK: Saved fare_model_metrics.json\n")

# ===============================================================
# ROUTE STATISTICS FOR WEB APP
# ===============================================================
print("=" * 60)
print("BUILDING ROUTE SUPPORT TABLES")
print("=" * 60)

route_stats = (
    raw_delays.filter(
        col("ORIGIN").isNotNull()
        & col("DEST").isNotNull()
        & col("DEP_DELAY").isNotNull()
    )
    .groupBy("ORIGIN", "DEST")
    .agg(
        spark_round(avg("DEP_DELAY"), 2).alias("hist_avg_dep_delay"),
        spark_round(avg("ARR_DELAY"), 2).alias("hist_avg_arr_delay"),
        count("*").alias("total_flights"),
    )
    .filter(col("total_flights") > 100)
    .withColumn(
        "delay_risk",
        when(col("hist_avg_dep_delay") < 10, "LOW")
        .when(col("hist_avg_dep_delay") < 25, "MEDIUM")
        .otherwise("HIGH"),
    )
)

total_routes = route_stats.count()
print(f"Route statistics computed for {total_routes:,} routes")
save_pandas_csv(route_stats, "route_statistics.csv")

fare_by_origin = (
    raw_fares.filter(
        col("ITIN_FARE").isNotNull()
        & (col("ITIN_FARE") > 10)
        & (col("ITIN_FARE") < 5000)
        & col("ORIGIN").isNotNull()
    )
    .groupBy("ORIGIN")
    .agg(
        spark_round(avg("ITIN_FARE"), 2).alias("hist_avg_fare"),
        spark_round(avg("ITIN_YIELD"), 4).alias("hist_avg_yield"),
    )
)

save_pandas_csv(fare_by_origin, "fare_by_origin.csv")

# ===============================================================
# ENSEMBLE RECOMMENDATION LAYER
# This combines the two model domains into one route-level score
# for Step 7. It is a weighted ensemble layer, not a third trainer.
# ===============================================================
print("=" * 60)
print("BUILDING ENSEMBLE RECOMMENDATION OUTPUT")
print("=" * 60)

ensemble_routes = (
    route_stats.join(fare_by_origin, on="ORIGIN", how="left")
    .withColumn(
        "delay_score",
        when(col("hist_avg_dep_delay") < 10, 1.0)
        .when(col("hist_avg_dep_delay") < 25, 0.5)
        .otherwise(0.0),
    )
    .withColumn(
        "fare_score",
        when(col("hist_avg_fare") < 300, 1.0)
        .when(col("hist_avg_fare") < 500, 0.5)
        .otherwise(0.0),
    )
    .withColumn(
        "volume_score",
        when(col("total_flights") >= 1000, 1.0)
        .when(col("total_flights") >= 300, 0.75)
        .otherwise(0.5),
    )
    .withColumn(
        "ensemble_score",
        spark_round(
            (col("delay_score") * 0.50)
            + (col("fare_score") * 0.35)
            + (col("volume_score") * 0.15),
            4,
        ),
    )
    .withColumn(
        "recommendation",
        when(col("ensemble_score") >= 0.75, "BUY")
        .when(col("ensemble_score") >= 0.45, "HOLD")
        .otherwise("AVOID"),
    )
    .orderBy(desc("ensemble_score"), desc("total_flights"))
)

save_pandas_csv(ensemble_routes, "ensemble_route_recommendations.csv")

ensemble_summary = {
    "ensemble_type": "weighted_route_recommendation_layer",
    "inputs": [
        "hist_avg_dep_delay",
        "hist_avg_arr_delay",
        "hist_avg_fare",
        "hist_avg_yield",
        "total_flights",
    ],
    "weights": {
        "delay_score": 0.50,
        "fare_score": 0.35,
        "volume_score": 0.15,
    },
    "recommendation_logic": {
        "BUY": "ensemble_score >= 0.75",
        "HOLD": "0.45 <= ensemble_score < 0.75",
        "AVOID": "ensemble_score < 0.45",
    },
}
with open(os.path.join(processed, "ensemble_model_summary.json"), "w") as file_obj:
    json.dump(ensemble_summary, file_obj, indent=2)
print("OK: Saved ensemble_model_summary.json")

thresholds = {
    "hist_avg_delay_min": 9.0,
    "hist_avg_fare_usd": 400.0,
    "low_delay_threshold": 10.0,
    "high_delay_threshold": 25.0,
    "cheap_fare_threshold": 300.0,
    "expensive_fare_threshold": 500.0,
    "recommendation_logic": {
        "BUY": "delay < 10 min AND fare below historical average",
        "HOLD": "delay 10-25 min OR fare near historical average",
        "AVOID": "delay > 25 min OR fare above $500",
    },
}
with open(os.path.join(processed, "recommendation_thresholds.json"), "w") as file_obj:
    json.dump(thresholds, file_obj, indent=2)
print("OK: Saved recommendation_thresholds.json\n")

# ===============================================================
# WRAP UP
# ===============================================================
print("=" * 60)
print("ALL MODELS AND FILES SAVED")
print("=" * 60)

print("\nModels:")
for folder in sorted(os.listdir(model_dir)):
    print(f"  OK {folder}")

print("\nSupport files:")
support_files = [
    "delay_feature_importance.csv",
    "delay_model_metrics.json",
    "fare_model_metrics.json",
    "route_statistics.csv",
    "fare_by_origin.csv",
    "ensemble_route_recommendations.csv",
    "ensemble_model_summary.json",
    "recommendation_thresholds.json",
]
for file_name in support_files:
    path_obj = os.path.join(processed, file_name)
    if os.path.exists(path_obj):
        print(f"  OK {file_name} ({os.path.getsize(path_obj):,} bytes)")
    else:
        print(f"  MISSING {file_name}")

print("\nMODEL SUMMARY")
print("Model 1 - Random Forest Delay Predictor")
print(f"  RMSE: {rmse_delay:.2f} min | MAE: {mae_delay:.2f} min | R2: {r2_delay:.4f}")
print("Model 2 - Gradient Boosted Price Predictor")
print(f"  RMSE: ${rmse_fare:.2f} | MAE: ${mae_fare:.2f} | R2: {r2_fare:.4f}")
print("Ensemble - Weighted Route Recommendation Layer")
print("  Output files: ensemble_route_recommendations.csv, ensemble_model_summary.json")

spark.stop()
print("\nOK: Step 5 complete")
print("Next: Run Step6_mongodb_load.py")
print("Then: Run Step7_webapp.py")