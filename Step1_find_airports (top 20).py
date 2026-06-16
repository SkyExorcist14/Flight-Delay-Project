import os
import glob
import sys

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Environment setup
# ══════════════════════════════════════════════════════════════════
java_home = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
os.environ["JAVA_HOME"]      = java_home
os.environ["HADOOP_HOME"]    = r"D:\hadoop"
os.environ["hadoop.home.dir"]= r"D:\hadoop"

print(f"✅ Java 17: {java_home}")
print(f"✅ Hadoop:  D:\\hadoop")

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Start Spark
# ══════════════════════════════════════════════════════════════════
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc

spark = SparkSession.builder \
    .appName("FindTopAirports") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark started!\n")

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Load CSVs using Python glob (bypasses hadoop scanner)
# ══════════════════════════════════════════════════════════════════
delay_folder = r"D:\UMBC All Data\DATA 603\flight_project\Data\Delays"

# Python finds the files — not Spark/Hadoop
csv_files = glob.glob(delay_folder + r"\*.csv")

if not csv_files:
    print(f"❌ No CSV files found in: {delay_folder}")
    print("   Check that your delay CSVs are in that folder")
    sys.exit(1)

print(f"✅ Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"   → {os.path.basename(f)}")

# Convert Windows backslash paths to forward slash for Spark
csv_files_spark = [f.replace("\\", "/") for f in csv_files]

print(f"\n⏳ Loading {len(csv_files_spark)} files into Spark...")
print("   This will take 5-10 minutes, please wait...\n")

# Pass the list directly — no wildcard, no hadoop file scanner
df = spark.read.csv(
    csv_files_spark,
    header=True,
    inferSchema=True
)

total = df.count()
print(f"✅ Total flight records loaded: {total:,}")

# Drop the empty junk column if it exists
if "Unnamed: 27" in df.columns:
    df = df.drop("Unnamed: 27")

print(f"✅ Columns: {df.columns}\n")

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Find Top 25 Airports
# ══════════════════════════════════════════════════════════════════
print("⏳ Counting flights per airport...")

departures = df.groupBy("ORIGIN") \
    .agg(count("*").alias("departure_count")) \
    .withColumnRenamed("ORIGIN", "airport")

arrivals = df.groupBy("DEST") \
    .agg(count("*").alias("arrival_count")) \
    .withColumnRenamed("DEST", "airport")

combined = departures.join(arrivals, on="airport", how="outer") \
    .fillna(0) \
    .withColumn("total_traffic",
                col("departure_count") + col("arrival_count")) \
    .orderBy(desc("total_traffic"))

top25 = combined.limit(25)

print("\n========== TOP 25 AIRPORTS BY TOTAL TRAFFIC ==========")
top25.show(25, truncate=False)

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Save Results
# ══════════════════════════════════════════════════════════════════
output_path = r"D:\UMBC All Data\DATA 603\flight_project\top_airports.csv"
top25.toPandas().to_csv(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")
print("📋 Open top_airports.csv in Excel — those are your top 25 airports!")
print("📋 Use the top 20 codes to download weather data from NOAA next.")

spark.stop()
print("✅ Done!")