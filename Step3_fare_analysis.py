import os
import glob
import sys
import pandas as pd
import json

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
fare_folder = r"D:\UMBC All Data\DATA 603\flight_project\Data\Fare"
processed   = r"D:\UMBC All Data\DATA 603\flight_project\Data\Processed"
os.makedirs(processed, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# STEP 1 — SCAN AND FIX CORRUPT FILES BEFORE STARTING SPARK
# Some BTS files store fares in cents instead of dollars
# If avg fare < $50 it means the data is in cents → multiply by 100
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("STEP 1: SCANNING FARE FILES FOR DATA QUALITY")
print("=" * 55)

all_csv = glob.glob(fare_folder + r"\*.csv")
print(f"Total files found: {len(all_csv)}\n")

corrupt_count = 0
clean_count   = 0

for filepath in all_csv:
    fname = os.path.basename(filepath)
    try:
        pdf = pd.read_csv(
            filepath,
            usecols=["YEAR", "QUARTER", "ITIN_FARE"],
            nrows=10000
        )
        pdf      = pdf[pdf["ITIN_FARE"].notna() & (pdf["ITIN_FARE"] > 0)]
        avg_fare = pdf["ITIN_FARE"].mean()
        year     = int(pdf["YEAR"].iloc[0])
        quarter  = int(pdf["QUARTER"].iloc[0])

        if avg_fare < 50:
            print(f"  ⚠️  Fixing {year} Q{quarter} | avg was ${avg_fare:.2f}")
            full = pd.read_csv(filepath)
            if "ITIN_FARE"  in full.columns:
                full["ITIN_FARE"]  = full["ITIN_FARE"]  * 100
            if "ITIN_YIELD" in full.columns:
                full["ITIN_YIELD"] = full["ITIN_YIELD"] * 100
            full.to_csv(filepath, index=False)
            avg_fixed = full["ITIN_FARE"][full["ITIN_FARE"] > 0].mean()
            print(f"     ✅ Fixed — avg is now ${avg_fixed:.2f}")
            corrupt_count += 1
        else:
            print(f"  ✅ {year} Q{quarter} | avg fare ${avg_fare:.2f}")
            clean_count += 1

    except Exception as e:
        print(f"  ❌ Could not read {fname}: {e}")

print(f"\n✅ Clean files  : {clean_count}")
print(f"✅ Fixed files  : {corrupt_count}")
print(f"Total processed : {clean_count + corrupt_count}\n")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — START SPARK
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("STEP 2: STARTING SPARK")
print("=" * 55)

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
    col, avg, count, desc,
    max as spark_max,
    min as spark_min,
    round as spark_round
)

spark = SparkSession.builder \
    .appName("FareAnalysis") \
    .config("spark.driver.memory",          "8g") \
    .config("spark.driver.maxResultSize",   "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.pyspark.python",          PYTHON_EXE) \
    .config("spark.pyspark.driver.python",   PYTHON_EXE) \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark started!\n")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — LOAD ALL FARE CSVs INTO SPARK
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("STEP 3: LOADING FARE DATA INTO SPARK")
print("=" * 55)

spark_paths = [f.replace("\\", "/") for f in all_csv]
print(f"Loading {len(spark_paths)} files...")

fare_df = spark.read.csv(
    spark_paths,
    header=True,
    inferSchema=True
)

needed   = ["YEAR", "QUARTER", "ORIGIN", "ORIGIN_STATE_ABR",
            "ITIN_FARE", "ITIN_YIELD", "PASSENGERS", "DISTANCE"]
existing = [c for c in needed if c in fare_df.columns]
fare_df  = fare_df.select(existing)

fare_df = fare_df.filter(
    col("ITIN_FARE").isNotNull() &
    (col("ITIN_FARE") > 10)     &
    (col("ITIN_FARE") < 5000)
)

total = fare_df.count()
print(f"✅ Total clean fare records: {total:,}")
print(f"✅ Columns: {fare_df.columns}\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Average fare by year
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 1: AVERAGE FARE BY YEAR")
print("=" * 55)

fare_by_year = fare_df.groupBy("YEAR") \
    .agg(
        spark_round(avg("ITIN_FARE"),       2).alias("avg_fare_usd"),
        spark_round(spark_min("ITIN_FARE"), 2).alias("min_fare_usd"),
        spark_round(spark_max("ITIN_FARE"), 2).alias("max_fare_usd"),
        count("*").alias("total_itineraries")
    ).orderBy("YEAR")

fare_by_year.show(20, truncate=False)
fare_by_year.toPandas().to_csv(
    f"{processed}/fare_by_year.csv", index=False)
print("✅ Saved fare_by_year.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Average fare by quarter
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 2: AVERAGE FARE BY QUARTER")
print("=" * 55)

fare_by_quarter = fare_df.groupBy("QUARTER") \
    .agg(
        spark_round(avg("ITIN_FARE"), 2).alias("avg_fare_usd"),
        count("*").alias("total_itineraries")
    ).orderBy("QUARTER")

fare_by_quarter.show(truncate=False)

qpdf = fare_by_quarter.toPandas()
qpdf["quarter_label"] = qpdf["QUARTER"].map({
    1: "Q1 Jan-Mar",
    2: "Q2 Apr-Jun",
    3: "Q3 Jul-Sep",
    4: "Q4 Oct-Dec"
})
qpdf.to_csv(f"{processed}/fare_by_quarter.csv", index=False)
print("✅ Saved fare_by_quarter.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Most expensive airports
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 3: TOP 20 MOST EXPENSIVE AIRPORTS")
print("=" * 55)

expensive = fare_df.groupBy("ORIGIN") \
    .agg(
        spark_round(avg("ITIN_FARE"),  2).alias("avg_fare_usd"),
        spark_round(avg("ITIN_YIELD"), 4).alias("avg_yield_per_mile"),
        count("*").alias("total_itineraries")
    ) \
    .filter(col("total_itineraries") > 500) \
    .orderBy(desc("avg_fare_usd"))

expensive.show(20, truncate=False)
expensive.toPandas().to_csv(
    f"{processed}/most_expensive_airports.csv", index=False)
print("✅ Saved most_expensive_airports.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Cheapest airports
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 4: TOP 20 CHEAPEST AIRPORTS")
print("=" * 55)

cheapest = fare_df.groupBy("ORIGIN") \
    .agg(
        spark_round(avg("ITIN_FARE"),  2).alias("avg_fare_usd"),
        spark_round(avg("ITIN_YIELD"), 4).alias("avg_yield_per_mile"),
        count("*").alias("total_itineraries")
    ) \
    .filter(col("total_itineraries") > 500) \
    .orderBy("avg_fare_usd")

cheapest.show(20, truncate=False)
cheapest.toPandas().to_csv(
    f"{processed}/cheapest_airports.csv", index=False)
print("✅ Saved cheapest_airports.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 5 — Price trend by year and quarter
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 5: PRICE TREND BY YEAR AND QUARTER")
print("=" * 55)

price_trend = fare_df.groupBy("YEAR", "QUARTER") \
    .agg(
        spark_round(avg("ITIN_FARE"), 2).alias("avg_fare_usd"),
        count("*").alias("total_itineraries")
    ).orderBy("YEAR", "QUARTER")

price_trend.show(44, truncate=False)
price_trend.toPandas().to_csv(
    f"{processed}/price_trend.csv", index=False)
print("✅ Saved price_trend.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 6 — Best value airports (price per mile)
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 6: BEST VALUE AIRPORTS (PRICE PER MILE)")
print("=" * 55)

best_value = fare_df.filter(
        col("DISTANCE").isNotNull() & (col("DISTANCE") > 0)
    ) \
    .groupBy("ORIGIN") \
    .agg(
        spark_round(avg("ITIN_FARE"),  2).alias("avg_fare_usd"),
        spark_round(avg("DISTANCE"),   0).alias("avg_distance_miles"),
        spark_round(avg("ITIN_YIELD"), 4).alias("avg_cents_per_mile"),
        count("*").alias("total_itineraries")
    ) \
    .filter(col("total_itineraries") > 500) \
    .orderBy("avg_cents_per_mile")

best_value.show(15, truncate=False)
best_value.toPandas().to_csv(
    f"{processed}/best_value_airports.csv", index=False)
print("✅ Saved best_value_airports.csv\n")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 7 — Overall summary for web app and ML model
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ANALYSIS 7: OVERALL FARE SUMMARY")
print("=" * 55)

summary = fare_df.agg(
    spark_round(avg("ITIN_FARE"),       2).alias("overall_avg_fare"),
    spark_round(spark_min("ITIN_FARE"), 2).alias("overall_min_fare"),
    spark_round(spark_max("ITIN_FARE"), 2).alias("overall_max_fare"),
    count("*").alias("total_records"),
).collect()[0]

print(f"  Overall average fare : ${summary['overall_avg_fare']}")
print(f"  Cheapest fare found  : ${summary['overall_min_fare']}")
print(f"  Most expensive fare  : ${summary['overall_max_fare']}")
print(f"  Total records        : {summary['total_records']:,}")

summary_data = {
    "overall_avg_fare" : float(summary["overall_avg_fare"]),
    "overall_min_fare" : float(summary["overall_min_fare"]),
    "overall_max_fare" : float(summary["overall_max_fare"]),
    "total_records"    : int(summary["total_records"]),
}
pd.DataFrame([summary_data]).to_csv(
    f"{processed}/fare_summary.csv", index=False)
with open(f"{processed}/fare_summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)
print("✅ Saved fare_summary.csv")
print("✅ Saved fare_summary.json\n")

# ══════════════════════════════════════════════════════════════════
# WRAP UP
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ALL FILES CREATED BY STEP 3:")
print("=" * 55)

step3_files = [
    "fare_by_year.csv",
    "fare_by_quarter.csv",
    "most_expensive_airports.csv",
    "cheapest_airports.csv",
    "price_trend.csv",
    "best_value_airports.csv",
    "fare_summary.csv",
    "fare_summary.json",
]

for f in step3_files:
    path = os.path.join(processed, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  ✅ {f}  ({size:,} bytes)")
    else:
        print(f"  ❌ {f}  NOT CREATED")

spark.stop()
print("\n✅ Step 3 complete!")
print("✅ Safe to run Step5_ml_model.py now")