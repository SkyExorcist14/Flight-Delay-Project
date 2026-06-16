import os
import sys
import json
import glob
import pandas as pd

# ================================================================
# DATA 603 - Big Data Project
# Step 6: Graph Analysis + Final Results JSON Generator
# ================================================================

PYTHON_EXE = r"C:\Python311\python.exe"

os.environ["PYSPARK_PYTHON"]        = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE
os.environ["JAVA_HOME"]             = r"C:\Program Files\EclipseAdoptium\jdk-17.0.18.8-hotspot"
os.environ["HADOOP_HOME"]           = r"D:\hadoop"
os.environ["hadoop.home.dir"]       = r"D:\hadoop"

if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

print(f"Using Python: {sys.executable}")
print(f"Python version: {sys.version}")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, desc, round as spark_round,
    when, sum as spark_sum
)
import pyspark.sql.functions as F

# ----------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------
PROJECT_ROOT = r"D:\UMBC All Data\DATA 603\flight_project"
processed    = rf"{PROJECT_ROOT}\Data\Processed"
delay_folder = rf"{PROJECT_ROOT}\Data\Delays"
output_json  = rf"{PROJECT_ROOT}\results.json"

# ----------------------------------------------------------------
# START SPARK
# ----------------------------------------------------------------
spark = SparkSession.builder \
    .appName("GraphAnalysis_Step6") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.pyspark.python",              PYTHON_EXE) \
    .config("spark.pyspark.driver.python",       PYTHON_EXE) \
    .config("spark.executorEnv.PYSPARK_PYTHON",  PYTHON_EXE) \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("Spark started\n")

# ================================================================
# SECTION 1 — AIRPORT RDD
# ================================================================
print("=" * 55)
print("SECTION 1: AIRPORT RDD")
print("=" * 55)

airports_data = [
    ("ATL", "Hartsfield-Jackson Atlanta",         "Georgia",       33.6407, -84.4277),
    ("ORD", "Chicago O'Hare International",        "Illinois",      41.9742, -87.9073),
    ("DFW", "Dallas/Fort Worth International",     "Texas",         32.8998, -97.0403),
    ("DEN", "Denver International",                "Colorado",      39.8561, -104.6737),
    ("LAX", "Los Angeles International",           "California",    33.9425, -118.4081),
    ("CLT", "Charlotte Douglas International",     "North Carolina",35.2140, -80.9431),
    ("LAS", "Harry Reid International",            "Nevada",        36.0840, -115.1537),
    ("PHX", "Phoenix Sky Harbor International",    "Arizona",       33.4373, -112.0078),
    ("MCO", "Orlando International",               "Florida",        28.4294, -81.3089),
    ("SEA", "Seattle-Tacoma International",        "Washington",    47.4502, -122.3088),
    ("MIA", "Miami International",                 "Florida",       25.7959, -80.2870),
    ("JFK", "John F. Kennedy International",       "New York",      40.6413, -73.7781),
    ("EWR", "Newark Liberty International",        "New Jersey",    40.6895, -74.1745),
    ("SFO", "San Francisco International",         "California",    37.6213, -122.3790),
    ("BOS", "Boston Logan International",          "Massachusetts", 42.3656, -71.0096),
    ("MSP", "Minneapolis-Saint Paul International","Minnesota",     44.8848, -93.2223),
    ("DTW", "Detroit Metropolitan",                "Michigan",      42.2162, -83.3554),
    ("PHL", "Philadelphia International",          "Pennsylvania",  39.8721, -75.2411),
    ("BWI", "Baltimore Washington International",  "Maryland",      39.1754, -76.6683),
    ("IAH", "George Bush Intercontinental",        "Texas",         29.9902, -95.3368),
    ("SLC", "Salt Lake City International",        "Utah",          40.7884, -111.9778),
    ("BNA", "Nashville International",             "Tennessee",     36.1245, -86.6782),
    ("MDW", "Chicago Midway International",        "Illinois",      41.7868, -87.7522),
    ("HOU", "William P. Hobby Airport",            "Texas",         29.6454, -95.2789),
]

airport_rdd = spark.sparkContext.parallelize(airports_data, 4)
print(f"Airport RDD created — {airport_rdd.count()} airports, {airport_rdd.getNumPartitions()} partitions")

codes_rdd = airport_rdd.map(lambda x: x[0])
print(f"Airport codes: {codes_rdd.collect()}")

texas_rdd = airport_rdd.filter(lambda x: x[2] == "Texas")
print(f"Texas airports: {[r[0] for r in texas_rdd.collect()]}")

# ================================================================
# SECTION 2 — LOAD DELAY DATA
# ================================================================
print("\n" + "=" * 55)
print("SECTION 2: LOADING DELAY DATA")
print("=" * 55)

delay_files = glob.glob(delay_folder + r"\*.csv")
delay_spark = [f.replace("\\", "/") for f in delay_files]
print(f"Loading {len(delay_spark)} delay files...")

df = spark.read.csv(delay_spark, header=True, inferSchema=True)
if "Unnamed: 27" in df.columns:
    df = df.drop("Unnamed: 27")

total = df.count()
print(f"Total records: {total:,}\n")

# ================================================================
# SECTION 3 — GRAPH / NETWORK ANALYSIS
# ================================================================
print("=" * 55)
print("SECTION 3: GRAPH / NETWORK ANALYSIS")
print("=" * 55)

outbound = df.groupBy("ORIGIN").agg(
    F.countDistinct("DEST").alias("outbound_routes"),
    count("*").alias("total_departures"),
    spark_round(avg("DEP_DELAY"), 2).alias("avg_dep_delay")
).withColumnRenamed("ORIGIN", "airport")

inbound = df.groupBy("DEST").agg(
    F.countDistinct("ORIGIN").alias("inbound_routes"),
    count("*").alias("total_arrivals"),
    spark_round(avg("ARR_DELAY"), 2).alias("avg_arr_delay")
).withColumnRenamed("DEST", "airport")

airport_nodes = outbound.join(inbound, on="airport", how="outer").fillna(0)
airport_nodes = airport_nodes.withColumn(
    "total_degree", col("outbound_routes") + col("inbound_routes")
).withColumn(
    "total_traffic", col("total_departures") + col("total_arrivals")
).withColumn(
    "network_importance_score",
    spark_round((col("total_traffic") / 1000000) * col("total_degree"), 2)
).orderBy(desc("total_traffic"))

print("TOP 20 AIRPORTS BY NETWORK DEGREE:")
airport_nodes.show(20, truncate=False)

route_edges = df.groupBy("ORIGIN", "DEST").agg(
    count("*").alias("flight_count"),
    spark_round(avg("DEP_DELAY"), 2).alias("avg_delay"),
    spark_round(avg("DISTANCE"),  0).alias("avg_distance")
).orderBy(desc("flight_count"))

print("TOP 20 BUSIEST ROUTES:")
route_edges.show(20, truncate=False)

# ================================================================
# SECTION 4 — ROUTE LOOKUP TABLE
# ================================================================
print("=" * 55)
print("SECTION 4: ROUTE LOOKUP TABLE")
print("=" * 55)

route_lookup = df.groupBy("ORIGIN", "DEST", "OP_CARRIER").agg(
    count("*").alias("total_flights"),
    spark_round(avg("DEP_DELAY"),     2).alias("avg_dep_delay_min"),
    spark_round(avg("ARR_DELAY"),     2).alias("avg_arr_delay_min"),
    spark_round(avg("WEATHER_DELAY"), 2).alias("avg_weather_delay"),
    spark_round(avg("CARRIER_DELAY"), 2).alias("avg_carrier_delay"),
    spark_round(
        count(when(col("DEP_DELAY") > 15, True)) / count("*") * 100, 2
    ).alias("pct_delayed_over_15min"),
    spark_round(
        count(when(col("CANCELLED") == 1, True)) / count("*") * 100, 2
    ).alias("cancellation_rate_pct")
).filter(col("total_flights") > 100)

print(f"Route combinations: {route_lookup.count():,}")

# ================================================================
# SECTION 5 — BUILD results.json
# ================================================================
print("\n" + "=" * 55)
print("SECTION 5: BUILDING results.json")
print("=" * 55)

def safe_read_csv(filename):
    path = os.path.join(processed, filename)
    if os.path.exists(path):
        return pd.read_csv(path).fillna(0).to_dict(orient="records")
    print(f"  WARNING: {filename} not found, skipping")
    return []

nodes_pd = airport_nodes.limit(24).toPandas().fillna(0)
airport_coords = {row[0]: {"lat": row[3], "lon": row[4], "state": row[2]} for row in airports_data}
nodes_list = []
for _, row in nodes_pd.iterrows():
    code   = str(row["airport"])
    coords = airport_coords.get(code, {"lat": 0, "lon": 0, "state": "Unknown"})
    nodes_list.append({
        "code":             code,
        "lat":              coords["lat"],
        "lon":              coords["lon"],
        "state":            coords["state"],
        "total_departures": int(row.get("total_departures", 0)),
        "total_arrivals":   int(row.get("total_arrivals",   0)),
        "total_traffic":    int(row.get("total_traffic",    0)),
        "outbound_routes":  int(row.get("outbound_routes",  0)),
        "avg_dep_delay":    float(row.get("avg_dep_delay",  0)),
        "avg_arr_delay":    float(row.get("avg_arr_delay",  0)),
        "network_score":    float(row.get("network_importance_score", 0)),
    })

edges_pd  = route_edges.limit(50).toPandas().fillna(0)
edges_list = [{
    "origin":       str(r["ORIGIN"]),
    "dest":         str(r["DEST"]),
    "flight_count": int(r["flight_count"]),
    "avg_delay":    float(r["avg_delay"]),
    "avg_distance": float(r["avg_distance"]),
} for _, r in edges_pd.iterrows()]

route_pd   = route_lookup.limit(5000).toPandas().fillna(0)
route_list = [{
    "origin":            str(r["ORIGIN"]),
    "dest":              str(r["DEST"]),
    "carrier":           str(r["OP_CARRIER"]),
    "total_flights":     int(r["total_flights"]),
    "avg_dep_delay":     float(r["avg_dep_delay_min"]),
    "avg_arr_delay":     float(r["avg_arr_delay_min"]),
    "avg_weather_delay": float(r["avg_weather_delay"]),
    "avg_carrier_delay": float(r["avg_carrier_delay"]),
    "pct_delayed":       float(r["pct_delayed_over_15min"]),
    "cancellation_rate": float(r["cancellation_rate_pct"]),
} for _, r in route_pd.iterrows()]

carrier_names = {
    "AA": "American Airlines", "DL": "Delta Air Lines",
    "UA": "United Airlines",   "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",   "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",   "F9": "Frontier Airlines",
    "G4": "Allegiant Air",     "SY": "Sun Country Airlines",
    "HA": "Hawaiian Airlines", "OO": "SkyWest Airlines",
    "YX": "Republic Airways",  "9E": "Endeavor Air",
    "MQ": "Envoy Air",         "OH": "PSA Airlines",
}

results = {
    "meta": {
        "generated":      pd.Timestamp.now().isoformat(),
        "total_airports": len(nodes_list),
        "total_routes":   len(route_list),
    },
    "airports":        nodes_list,
    "busiest_routes":  edges_list,
    "route_lookup":    route_list,
    "carrier_names":   carrier_names,
    "fare_by_year":    safe_read_csv("fare_by_year.csv"),
    "fare_by_quarter": safe_read_csv("fare_by_quarter.csv"),
    "monthly_delays":  safe_read_csv("monthly_delays.csv"),
    "daily_delays":    safe_read_csv("daily_delays.csv"),
    "delay_causes":    safe_read_csv("delay_causes.csv"),
    "most_expensive":  safe_read_csv("most_expensive_airports.csv"),
    "cheapest":        safe_read_csv("cheapest_airports.csv"),
    "best_value":      safe_read_csv("best_value_airports.csv"),
    "worst_weather":   safe_read_csv("worst_weather_airports.csv"),
    "top_dep_delays":  safe_read_csv("top_dep_delays.csv"),
    "top_arr_delays":  safe_read_csv("top_arr_delays.csv"),
}

with open(output_json, "w") as f:
    json.dump(results, f, indent=2)

size_mb = os.path.getsize(output_json) / (1024 * 1024)
print(f"\nresults.json saved to: {output_json}")
print(f"File size: {size_mb:.2f} MB")
print(f"Airports : {len(nodes_list)}")
print(f"Routes   : {len(route_list):,}")

spark.stop()
print("\nStep 6 complete! Now open webapp.html in Chrome.")