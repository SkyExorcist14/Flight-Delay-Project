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

# Remove any reference to Spark 4.x folder
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

if not os.path.exists(PYTHON_EXE):
    print(f"❌ Python not found at: {PYTHON_EXE}")
    sys.exit(1)
else:
    print(f"✅ Python found at: {PYTHON_EXE}")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, desc,
    sum as spark_sum, year, month,
    dayofweek, when, round as spark_round
)
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd

# Verify pyspark version — must be 3.5.x
import pyspark
print(f"✅ PySpark version: {pyspark.__version__}")
if pyspark.__version__.startswith("4"):
    print("❌ You have PySpark 4.x — RDDs won't work. Run:")
    print('   pip uninstall pyspark -y && pip install pyspark==3.5.1')
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════
# START SPARK
# ══════════════════════════════════════════════════════════════════
spark = SparkSession.builder \
    .appName("DelayAnalysis") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.pyspark.python",               PYTHON_EXE) \
    .config("spark.pyspark.driver.python",        PYTHON_EXE) \
    .config("spark.python.worker.faulthandler.enabled", "true") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark started!\n")

# ══════════════════════════════════════════════════════════════════
# REQUIREMENT 1 — RDD OF US AIRPORTS
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("REQUIREMENT 1: RDD OF US AIRPORTS")
print("=" * 55)

airports_data = [
    ("ATL", "Hartsfield-Jackson Atlanta International", "Georgia"),
    ("ORD", "Chicago O'Hare International",             "Illinois"),
    ("DFW", "Dallas/Fort Worth International",          "Texas"),
    ("DEN", "Denver International Airport",             "Colorado"),
    ("LAX", "Los Angeles International",                "California"),
    ("CLT", "Charlotte Douglas International",          "North Carolina"),
    ("LAS", "Harry Reid International",                 "Nevada"),
    ("PHX", "Phoenix Sky Harbor International",         "Arizona"),
    ("MCO", "Orlando International Airport",            "Florida"),
    ("SEA", "Seattle-Tacoma International",             "Washington"),
    ("MIA", "Miami International Airport",              "Florida"),
    ("JFK", "John F. Kennedy International",            "New York"),
    ("EWR", "Newark Liberty International",             "New Jersey"),
    ("SFO", "San Francisco International",              "California"),
    ("BOS", "Boston Logan International",               "Massachusetts"),
    ("MSP", "Minneapolis-Saint Paul International",     "Minnesota"),
    ("DTW", "Detroit Metropolitan Wayne County",        "Michigan"),
    ("PHL", "Philadelphia International",               "Pennsylvania"),
    ("BWI", "Baltimore Washington International",       "Maryland"),
    ("IAD", "Washington Dulles International",          "Virginia"),
    ("HOU", "William P. Hobby Airport",                 "Texas"),
    ("IAH", "George Bush Intercontinental",             "Texas"),
    ("SLC", "Salt Lake City International",             "Utah"),
    ("BNA", "Nashville International Airport",          "Tennessee"),
]

airport_rdd = spark.sparkContext.parallelize(airports_data, 4)
num_airports = airport_rdd.count()
print(f"✅ Airport RDD created with {num_airports} airports")
print(f"✅ Partitions: {airport_rdd.getNumPartitions()}")

codes_rdd = airport_rdd.map(lambda x: x[0])
print(f"\n✅ Airport codes (map):")
print(codes_rdd.collect())

texas_rdd = airport_rdd.filter(lambda x: x[2] == "Texas")
print(f"\n✅ Texas airports (filter):")
for row in texas_rdd.collect():
    print(f"   {row[0]} — {row[1]}")

kv_rdd = airport_rdd.map(lambda x: (x[0], x[1]))
print(f"\n✅ Key-Value pairs (first 5):")
for k, v in kv_rdd.take(5):
    print(f"   {k} → {v}")

state_rdd = airport_rdd.map(lambda x: (x[2], x[0])) \
                        .groupByKey() \
                        .mapValues(list)
print(f"\n✅ Grouped by state:")
for state, codes in sorted(state_rdd.collect()):
    print(f"   {state}: {codes}")

schema = StructType([
    StructField("code",  StringType(), True),
    StructField("name",  StringType(), True),
    StructField("state", StringType(), True),
])
airport_df = spark.createDataFrame(airport_rdd, schema=schema)
print(f"\n✅ RDD → DataFrame:")
airport_df.show(25, truncate=False)

processed = r"D:\UMBC All Data\DATA 603\flight_project\Data\Processed"
os.makedirs(processed, exist_ok=True)
airport_df.toPandas().to_csv(f"{processed}/airports_list.csv", index=False)
print("✅ Saved airports_list.csv\n")

# ══════════════════════════════════════════════════════════════════
# LOAD DELAY DATA
# ══════════════════════════════════════════════════════════════════
delay_folder    = r"D:\UMBC All Data\DATA 603\flight_project\Data\Delays"
csv_files       = glob.glob(delay_folder + r"\*.csv")
csv_files_spark = [f.replace("\\", "/") for f in csv_files]

print(f"⏳ Loading {len(csv_files_spark)} delay CSV files...")
df = spark.read.csv(csv_files_spark, header=True, inferSchema=True)

if "Unnamed: 27" in df.columns:
    df = df.drop("Unnamed: 27")

df = df.withColumn("year",        year(col("FL_DATE"))) \
       .withColumn("month",       month(col("FL_DATE"))) \
       .withColumn("day_of_week", dayofweek(col("FL_DATE")))

total = df.count()
print(f"✅ Total records loaded: {total:,}\n")

# ══════════════════════════════════════════════════════════════════
# REQUIREMENTS 2–6 + BONUSES
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("REQUIREMENT 2: AIRPORT WITH MOST DEPARTURES")
print("=" * 55)
most_departures = df.groupBy("ORIGIN") \
    .agg(count("*").alias("total_departures")) \
    .orderBy(desc("total_departures"))
most_departures.show(10, truncate=False)
most_departures.toPandas().to_csv(f"{processed}/most_departures.csv", index=False)
print("✅ Saved\n")

print("=" * 55)
print("REQUIREMENT 3: AIRPORT WITH MOST ARRIVALS")
print("=" * 55)
most_arrivals = df.groupBy("DEST") \
    .agg(count("*").alias("total_arrivals")) \
    .orderBy(desc("total_arrivals"))
most_arrivals.show(10, truncate=False)
most_arrivals.toPandas().to_csv(f"{processed}/most_arrivals.csv", index=False)
print("✅ Saved\n")

print("=" * 55)
print("REQUIREMENT 4: BUSIEST AIRPORTS")
print("=" * 55)
dep_counts = df.groupBy("ORIGIN").agg(count("*").alias("flights")).withColumnRenamed("ORIGIN","airport")
arr_counts = df.groupBy("DEST").agg(count("*").alias("flights")).withColumnRenamed("DEST","airport")
busiest = dep_counts.union(arr_counts) \
    .groupBy("airport").agg(spark_sum("flights").alias("total_traffic")) \
    .orderBy(desc("total_traffic"))
busiest.show(15, truncate=False)
busiest.toPandas().to_csv(f"{processed}/busiest_airports.csv", index=False)
print("✅ Saved\n")

print("=" * 55)
print("REQUIREMENT 5: TOP 10 DEPARTURE DELAY AIRPORTS")
print("=" * 55)
top_dep_delays = df.filter(col("DEP_DELAY").isNotNull()) \
    .groupBy("ORIGIN") \
    .agg(
        spark_round(avg("DEP_DELAY"),2).alias("avg_dep_delay_min"),
        count("*").alias("total_flights"),
        spark_round(count(when(col("DEP_DELAY")>15,True))/count("*")*100,2).alias("pct_over_15min")
    ) \
    .filter(col("total_flights") > 10000) \
    .orderBy(desc("avg_dep_delay_min"))
top_dep_delays.show(10, truncate=False)
top_dep_delays.toPandas().to_csv(f"{processed}/top_dep_delays.csv", index=False)
print("✅ Saved\n")

print("=" * 55)
print("REQUIREMENT 6: TOP 10 ARRIVAL DELAY AIRPORTS")
print("=" * 55)
top_arr_delays = df.filter(col("ARR_DELAY").isNotNull()) \
    .groupBy("DEST") \
    .agg(
        spark_round(avg("ARR_DELAY"),2).alias("avg_arr_delay_min"),
        count("*").alias("total_flights"),
        spark_round(count(when(col("ARR_DELAY")>15,True))/count("*")*100,2).alias("pct_over_15min")
    ) \
    .filter(col("total_flights") > 10000) \
    .orderBy(desc("avg_arr_delay_min"))
top_arr_delays.show(10, truncate=False)
top_arr_delays.toPandas().to_csv(f"{processed}/top_arr_delays.csv", index=False)
print("✅ Saved\n")

print("=" * 55)
print("BONUS: DELAY BY CAUSE")
print("=" * 55)
delay_causes = df.agg(
    spark_round(avg("CARRIER_DELAY"),2).alias("avg_carrier_delay_min"),
    spark_round(avg("WEATHER_DELAY"),2).alias("avg_weather_delay_min"),
    spark_round(avg("NAS_DELAY"),2).alias("avg_nas_delay_min"),
    spark_round(avg("SECURITY_DELAY"),2).alias("avg_security_delay_min"),
    spark_round(avg("LATE_AIRCRAFT_DELAY"),2).alias("avg_late_aircraft_min"),
)
delay_causes.show(truncate=False)
delay_causes.toPandas().to_csv(f"{processed}/delay_causes.csv", index=False)
print("✅ Saved\n")

print("=" * 55)
print("BONUS: DELAY BY MONTH")
print("=" * 55)
monthly_delays = df.filter(col("DEP_DELAY").isNotNull()) \
    .groupBy("month").agg(spark_round(avg("DEP_DELAY"),2).alias("avg_dep_delay")) \
    .orderBy("month")
monthly_delays.show(12)
monthly_delays.toPandas().to_csv(f"{processed}/monthly_delays.csv", index=False)
print("✅ Saved\n")

print("=" * 55)
print("BONUS: DELAY BY DAY OF WEEK")
print("=" * 55)
daily_delays = df.filter(col("DEP_DELAY").isNotNull()) \
    .groupBy("day_of_week").agg(spark_round(avg("DEP_DELAY"),2).alias("avg_dep_delay")) \
    .orderBy("day_of_week")
daily_delays.show(7)
daily_delays.toPandas().to_csv(f"{processed}/daily_delays.csv", index=False)
print("✅ Saved\n")

print("=" * 55)
print("BONUS: CANCELLATION RATE BY AIRLINE")
print("=" * 55)
cancellations = df.groupBy("OP_CARRIER") \
    .agg(
        spark_round(avg(col("CANCELLED").cast("double"))*100,2).alias("cancellation_rate_pct"),
        count("*").alias("total_flights")
    ) \
    .filter(col("total_flights") > 50000) \
    .orderBy(desc("cancellation_rate_pct"))
cancellations.show(15)
cancellations.toPandas().to_csv(f"{processed}/cancellation_rates.csv", index=False)
print("✅ Saved\n")

# ══════════════════════════════════════════════════════════════════
# WRAP UP
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("ALL FILES SAVED:")
for f in sorted(os.listdir(processed)):
    size = os.path.getsize(os.path.join(processed, f))
    print(f"  ✅ {f}  ({size:,} bytes)")

spark.stop()
print("\n✅ Step 2 complete!")