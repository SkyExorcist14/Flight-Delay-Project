import os
import glob
import sys

# ══════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════
PYTHON_EXE = r"C:\Python311\python.exe"

os.environ["JAVA_HOME"]             = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
os.environ["HADOOP_HOME"]           = r"D:\hadoop"
os.environ["hadoop.home.dir"]       = r"D:\hadoop"
os.environ["PYSPARK_PYTHON"]        = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE

if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, desc, max as spark_max,
    min as spark_min, round as spark_round,
    when, isnan, year, month, to_date, corr
)
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
weather_folder = r"D:\UMBC All Data\DATA 603\flight_project\Data\Weather"
delay_folder   = r"D:\UMBC All Data\DATA 603\flight_project\Data\Delays"
processed      = r"D:\UMBC All Data\DATA 603\flight_project\Data\Processed"
os.makedirs(processed, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# START SPARK
# ══════════════════════════════════════════════════════════════════
spark = SparkSession.builder \
    .appName("WeatherAnalysis") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.pyspark.python",        PYTHON_EXE) \
    .config("spark.pyspark.driver.python", PYTHON_EXE) \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark started!\n")

# ══════════════════════════════════════════════════════════════════
# LOAD WEATHER DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("LOADING WEATHER DATA")
print("=" * 55)

weather_files = glob.glob(weather_folder + r"\*.csv")
weather_spark = [f.replace("\\", "/") for f in weather_files]

print(f"Found {len(weather_files)} weather files")

weather_df = spark.read.csv(
    weather_spark,
    header=True,
    inferSchema=True
)

# Keep only useful columns
weather_cols = ["STATION","NAME","DATE","AWND","PRCP","SNOW","SNWD","TMAX","TMIN","TAVG"]
existing_weather_cols = [c for c in weather_cols if c in weather_df.columns]
weather_df = weather_df.select(existing_weather_cols)

total_weather = weather_df.count()
print(f"✅ Weather records loaded: {total_weather:,}")
print(f"✅ Columns: {weather_df.columns}\n")

# ══════════════════════════════════════════════════════════════════
# EXTRACT AIRPORT CODE FROM NAME COLUMN
# The NAME column has full airport name — we extract the
# city/state part to match with our airport codes
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("WEATHER SUMMARY BY STATION")
print("=" * 55)

weather_summary = weather_df.groupBy("NAME") \
    .agg(
        spark_round(avg("PRCP"), 4).alias("avg_daily_precipitation"),
        spark_round(avg("SNOW"), 4).alias("avg_daily_snow"),
        spark_round(avg("TMAX"), 2).alias("avg_max_temp"),
        spark_round(avg("TMIN"), 2).alias("avg_min_temp"),
        spark_round(avg("AWND"), 2).alias("avg_wind_speed"),
        count("*").alias("days_recorded")
    ) \
    .orderBy(desc("avg_daily_precipitation"))

weather_summary.show(25, truncate=False)
weather_summary.toPandas().to_csv(
    f"{processed}/weather_by_station.csv", index=False)
print("✅ Saved weather_by_station.csv\n")

# ══════════════════════════════════════════════════════════════════
# LOAD DELAY DATA FOR CORRELATION
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("LOADING DELAY DATA FOR WEATHER CORRELATION")
print("=" * 55)

delay_files       = glob.glob(delay_folder + r"\*.csv")
delay_files_spark = [f.replace("\\", "/") for f in delay_files]

print(f"⏳ Loading {len(delay_files_spark)} delay files...")

delay_df = spark.read.csv(
    delay_files_spark,
    header=True,
    inferSchema=True
)

if "Unnamed: 27" in delay_df.columns:
    delay_df = delay_df.drop("Unnamed: 27")

delay_df = delay_df.withColumn("year",  year(col("FL_DATE"))) \
                   .withColumn("month", month(col("FL_DATE")))

total_delays = delay_df.count()
print(f"✅ Delay records loaded: {total_delays:,}\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Weather delay statistics
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 1: WEATHER DELAY STATISTICS BY AIRPORT")
print("=" * 55)

weather_delays = delay_df.filter(
        col("WEATHER_DELAY").isNotNull() &
        (col("WEATHER_DELAY") > 0)
    ) \
    .groupBy("ORIGIN") \
    .agg(
        spark_round(avg("WEATHER_DELAY"), 2).alias("avg_weather_delay_min"),
        count("*").alias("weather_delayed_flights"),
        spark_round(spark_max("WEATHER_DELAY"), 2).alias("max_weather_delay_min")
    ) \
    .filter(col("weather_delayed_flights") > 100) \
    .orderBy(desc("avg_weather_delay_min"))

print("TOP 20 AIRPORTS MOST AFFECTED BY WEATHER DELAYS:")
weather_delays.show(20, truncate=False)
weather_delays.toPandas().to_csv(
    f"{processed}/weather_delays_by_airport.csv", index=False)
print("✅ Saved weather_delays_by_airport.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Weather delay by month
# Shows which months have most weather-related disruptions
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 2: WEATHER DELAYS BY MONTH")
print("=" * 55)

weather_by_month = delay_df.filter(col("WEATHER_DELAY").isNotNull()) \
    .groupBy("month") \
    .agg(
        spark_round(avg("WEATHER_DELAY"), 2).alias("avg_weather_delay"),
        spark_round(avg("DEP_DELAY"), 2).alias("avg_total_delay"),
        count("*").alias("total_flights")
    ) \
    .orderBy("month")

weather_by_month.show(12)
weather_by_month.toPandas().to_csv(
    f"{processed}/weather_delays_by_month.csv", index=False)
print("✅ Saved weather_delays_by_month.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Worst weather airports for the web app
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 3: WORST WEATHER AIRPORTS (FOR WEB APP)")
print("=" * 55)

worst_weather_airports = delay_df \
    .groupBy("ORIGIN") \
    .agg(
        spark_round(avg("WEATHER_DELAY"), 2).alias("avg_weather_delay_min"),
        spark_round(
            count(when(col("WEATHER_DELAY") > 30, True)) /
            count("*") * 100, 2
        ).alias("pct_severe_weather_delay"),
        count("*").alias("total_flights")
    ) \
    .filter(col("total_flights") > 10000) \
    .orderBy(desc("avg_weather_delay_min"))

worst_weather_airports.show(15, truncate=False)
worst_weather_airports.toPandas().to_csv(
    f"{processed}/worst_weather_airports.csv", index=False)
print("✅ Saved worst_weather_airports.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Delay type breakdown (% weather vs carrier etc)
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 4: DELAY TYPE BREAKDOWN PERCENTAGE")
print("=" * 55)

total_flights = delay_df.count()

delay_breakdown = delay_df.agg(
    spark_round(
        count(when(col("WEATHER_DELAY") > 0, True)) / total_flights * 100, 2
    ).alias("pct_weather_caused"),
    spark_round(
        count(when(col("CARRIER_DELAY") > 0, True)) / total_flights * 100, 2
    ).alias("pct_carrier_caused"),
    spark_round(
        count(when(col("NAS_DELAY") > 0, True)) / total_flights * 100, 2
    ).alias("pct_nas_caused"),
    spark_round(
        count(when(col("LATE_AIRCRAFT_DELAY") > 0, True)) / total_flights * 100, 2
    ).alias("pct_late_aircraft_caused"),
    spark_round(
        count(when(col("SECURITY_DELAY") > 0, True)) / total_flights * 100, 2
    ).alias("pct_security_caused"),
)

delay_breakdown.show(truncate=False)
delay_breakdown.toPandas().to_csv(
    f"{processed}/delay_type_breakdown.csv", index=False)
print("✅ Saved delay_type_breakdown.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 5 — Seasonal weather patterns from NOAA data
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 5: SEASONAL WEATHER PATTERNS")
print("=" * 55)

weather_with_month = weather_df.withColumn(
    "month", month(col("DATE").cast("date"))
)

seasonal = weather_with_month.groupBy("month") \
    .agg(
        spark_round(avg("PRCP"), 4).alias("avg_precipitation"),
        spark_round(avg("SNOW"), 4).alias("avg_snowfall"),
        spark_round(avg("TMAX"), 2).alias("avg_max_temp"),
        spark_round(avg("AWND"), 2).alias("avg_wind_speed"),
    ) \
    .orderBy("month")

seasonal.show(12)
seasonal.toPandas().to_csv(
    f"{processed}/seasonal_weather.csv", index=False)
print("✅ Saved seasonal_weather.csv\n")

# ══════════════════════════════════════════════════════════════════
# WRAP UP
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ALL WEATHER FILES SAVED:")
weather_result_files = [
    f for f in sorted(os.listdir(processed))
    if "weather" in f or "seasonal" in f or "delay_type" in f
]
for f in weather_result_files:
    size = os.path.getsize(os.path.join(processed, f))
    print(f"  ✅ {f}  ({size:,} bytes)")

spark.stop()
print("\n✅ Step 4 complete! Run Step5_ml_model.py next.")