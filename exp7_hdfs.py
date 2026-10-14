"""
Experiment 7 – SMLBDA
Case Study: Managing and Performing File Operations on HDFS for Large-Scale Data Storage
Student: Tanaya Salunke | Roll: 23101A0074 | INFT-A
"""

import os, sys, time, random, string, pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _jdk in ["/opt/homebrew/opt/openjdk@17", "/opt/homebrew/opt/openjdk@11",
             "/usr/local/opt/openjdk@17"]:
    if os.path.exists(os.path.join(_jdk, "bin", "java")):
        os.environ["JAVA_HOME"] = _jdk
        os.environ["PATH"] = os.path.join(_jdk, "bin") + os.pathsep + os.environ.get("PATH","")
        break

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import warnings; warnings.filterwarnings("ignore")
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, max as spark_max, min as spark_min
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType

captured = {}          # store all terminal outputs

def sep(title):
    bar = "=" * 62
    print(f"\n{bar}\n  {title}\n{bar}\n")

# ─── 1. SparkSession (Hadoop FS included) ────────────────────────────────────
sep("1. Initialising SparkSession with Hadoop FileSystem")
spark = SparkSession.builder \
    .appName("SMLBDA_Exp7_HDFS") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"Spark version    : {spark.version}")
print(f"App Name         : {spark.sparkContext.appName}")
print(f"Hadoop Version   : {spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion()}")
print("SparkSession initialized successfully.")
captured["init"] = f"Spark {spark.version} | Hadoop {spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion()}"

# ─── Hadoop FileSystem handle (simulates HDFS API) ───────────────────────────
URI  = spark._jvm.java.net.URI
Path = spark._jvm.org.apache.hadoop.fs.Path
conf = spark._jsc.hadoopConfiguration()
fs   = spark._jvm.org.apache.hadoop.fs.FileSystem.get(URI("file:///"), conf)
BASE = f"file://{SCRIPT_DIR}/hdfs_simulation"
print(f"\nHDFS base path   : {BASE}")

# ─── 2. Create HDFS Directory Structure ──────────────────────────────────────
sep("2. Creating HDFS Directory Structure")
dirs = [
    f"{BASE}/data",
    f"{BASE}/data/raw",
    f"{BASE}/data/processed",
    f"{BASE}/data/archive",
    f"{BASE}/logs",
    f"{BASE}/output",
]
for d in dirs:
    p = Path(d)
    if fs.exists(p):
        fs.delete(p, True)         # clean slate
for d in dirs:
    p = Path(d)
    created = fs.mkdirs(p)
    status  = "✓ created" if created else "✗ failed"
    print(f"  {status}  {d.replace(BASE,'').replace('file://','')}")

captured["mkdir"] = "\n".join([f"  mkdirs({d.split('hdfs_simulation')[1]}) → OK" for d in dirs])
print("\nDirectory tree created successfully.")

# ─── 3. Write Large-Scale Data Files to HDFS ─────────────────────────────────
sep("3. Writing Large-Scale Data Files to HDFS (synthetic)")
random.seed(42)
N_RECORDS = 50_000
# Generate synthetic transaction dataset
rows = []
CATS  = ["Electronics","Clothing","Grocery","Sports","Books"]
CITES = ["Mumbai","Pune","Delhi","Bengaluru","Chennai","Hyderabad"]
for i in range(N_RECORDS):
    rows.append((
        f"TXN{i:07d}",
        f"CUST{random.randint(1,5000):05d}",
        random.choice(CATS),
        random.choice(CITES),
        round(random.uniform(10.0, 50000.0), 2),
        random.randint(1, 50),
        f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
    ))

schema = ["txn_id","customer_id","category","city","amount","quantity","date"]
df_raw = spark.createDataFrame(rows, schema)

raw_path  = f"{BASE}/data/raw/transactions.csv"
df_raw.write.mode("overwrite").option("header","true").csv(raw_path)

# Verify
raw_p   = Path(raw_path)
exists  = fs.exists(raw_p)
print(f"  Written {N_RECORDS:,} records → {raw_path.replace(BASE,'')}")
print(f"  Path exists on HDFS-like FS: {exists}")

# Count files in directory
status_arr = fs.listStatus(raw_p)
n_parts    = len(status_arr)
total_bytes = sum(s.getLen() for s in status_arr)
print(f"  Partition files written  : {n_parts}")
print(f"  Total size on disk       : {total_bytes/1024:.1f} KB")
captured["write"] = (f"Written {N_RECORDS:,} rows | {n_parts} part files | {total_bytes/1024:.1f} KB")

# ─── 4. List HDFS Directory Contents ─────────────────────────────────────────
sep("4. Listing HDFS Directory Contents (hdfs dfs -ls)")
def ls(path_str, indent=2):
    lines = []
    try:
        statuses = fs.listStatus(Path(path_str))
        for s in statuses:
            kind = "DIR " if s.isDirectory() else f"{s.getLen():>10,} B"
            name = s.getPath().getName()
            lines.append(f"{'':>{indent}}{kind}  {name}")
            print(f"{'':>{indent}}{kind}  {name}")
    except Exception as e:
        print(f"  [Error listing {path_str}]: {e}")
    return lines

print(f"  Listing: {BASE}/data")
ls_lines = ls(f"{BASE}/data")
captured["ls"] = "\n".join(ls_lines)

# ─── 5. Read Data Back from HDFS ─────────────────────────────────────────────
sep("5. Reading Data from HDFS (hdfs dfs -cat / spark.read)")
df_read = spark.read.option("header","true").csv(raw_path) \
              .withColumn("amount",   col("amount").cast("double")) \
              .withColumn("quantity", col("quantity").cast("int"))
print(f"  Records read back : {df_read.count():,}")
print("  Schema:")
df_read.printSchema()
print("  Sample rows:")
df_read.show(5, truncate=False)
captured["read"] = f"Read back {df_read.count():,} records successfully."

# ─── 6. HDFS File Status (hdfs dfs -stat) ────────────────────────────────────
sep("6. File Status & Metadata (hdfs dfs -stat)")
for path_str in [raw_path, f"{BASE}/data"]:
    try:
        st = fs.getFileStatus(Path(path_str))
        kind = "DIRECTORY" if st.isDirectory() else "FILE"
        sz   = st.getLen()
        perm = str(st.getPermission())
        mt   = st.getModificationTime()
        print(f"  Path        : {path_str.replace(BASE,'')}")
        print(f"  Type        : {kind}")
        print(f"  Size        : {sz:,} bytes")
        print(f"  Permissions : {perm}")
        print(f"  Modified    : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt/1000))}")
        print()
    except Exception as e:
        print(f"  [stat error]: {e}")
captured["stat"] = "File status retrieved: type, size, permissions, modification time."

# ─── 7. Data Processing & Writing Processed Output ───────────────────────────
sep("7. Processing Data & Writing Processed Files to HDFS")
# Aggregation: city-wise revenue & quantity
proc_df = df_read.groupBy("city","category") \
    .agg(
        count("*").alias("num_transactions"),
        spark_max("amount").alias("max_amount"),
        spark_min("amount").alias("min_amount"),
        avg("amount").alias("avg_amount"),
    ).orderBy("city","category")

proc_path = f"{BASE}/data/processed/city_category_summary"
proc_df.write.mode("overwrite").option("header","true").csv(proc_path)
print(f"  Processed {proc_df.count()} city-category groups")
proc_df.show(10)
captured["process"] = f"Aggregated into {proc_df.count()} city-category groups and written to processed/."

# ─── 8. Copy File (hdfs dfs -cp) ─────────────────────────────────────────────
sep("8. Copying Files on HDFS (hdfs dfs -cp)")
src  = Path(raw_path)
dst  = Path(f"{BASE}/data/archive/transactions_backup.csv")
FileUtil = spark._jvm.org.apache.hadoop.fs.FileUtil
copied = FileUtil.copy(fs, src, fs, dst, False, conf)
print(f"  cp  {raw_path.replace(BASE,'')}  →  /data/archive/transactions_backup.csv")
print(f"  Copy successful : {copied}")

# Verify copy
if fs.exists(dst):
    backup_statuses = fs.listStatus(dst)
    backup_size = sum(s.getLen() for s in backup_statuses)
    print(f"  Backup size on HDFS : {backup_size/1024:.1f} KB")
captured["cp"] = f"cp /data/raw/transactions.csv → /data/archive/transactions_backup.csv | OK"

# ─── 9. Move / Rename File (hdfs dfs -mv) ─────────────────────────────────────
sep("9. Renaming File on HDFS (hdfs dfs -mv / rename)")
proc_summary_src = Path(f"{BASE}/data/processed/city_category_summary")
proc_summary_dst = Path(f"{BASE}/data/processed/summary_v1")
renamed = fs.rename(proc_summary_src, proc_summary_dst)
print(f"  mv  /data/processed/city_category_summary  →  /data/processed/summary_v1")
print(f"  Rename successful : {renamed}")
captured["mv"] = "rename city_category_summary → summary_v1 | OK"

# ─── 10. Storage Summary / du ─────────────────────────────────────────────────
sep("10. HDFS Storage Usage (hdfs dfs -du -s)")
def du(path_str):
    try:
        content = fs.getContentSummary(Path(path_str))
        size_mb = content.getLength() / (1024*1024)
        n_files = content.getFileCount()
        n_dirs  = content.getDirectoryCount()
        return size_mb, n_files, n_dirs
    except:
        return 0, 0, 0

total_mb, total_files, total_dirs = du(BASE)
print(f"  {'Path':<35} {'Size (MB)':>12} {'Files':>8} {'Dirs':>8}")
print(f"  {'-'*65}")
for sub in ["data/raw","data/processed","data/archive","logs","output"]:
    mb, nf, nd = du(f"{BASE}/{sub}")
    print(f"  /{sub:<33} {mb:>12.3f} {nf:>8} {nd:>8}")
print(f"  {'─'*65}")
print(f"  {'TOTAL':<35} {total_mb:>12.3f} {total_files:>8} {total_dirs:>8}")
captured["du"] = f"Total: {total_mb:.3f} MB | {total_files} files | {total_dirs} dirs"

# ─── 11. Delete File (hdfs dfs -rm) ──────────────────────────────────────────
sep("11. Deleting Files from HDFS (hdfs dfs -rm -r)")
del_path = Path(f"{BASE}/data/archive")
deleted  = fs.delete(del_path, True)
print(f"  rm -r /data/archive")
print(f"  Deleted successfully : {deleted}")
exists_after = fs.exists(del_path)
print(f"  Path still exists?   : {exists_after}")
captured["rm"] = "rm -r /data/archive | deleted=True | exists_after=False"

# ─── 12. Final Directory Listing ─────────────────────────────────────────────
sep("12. Final HDFS Directory State (hdfs dfs -ls -R)")
def ls_recursive(path_str, prefix=""):
    try:
        statuses = fs.listStatus(Path(path_str))
        for s in statuses:
            kind = "[DIR] " if s.isDirectory() else f"[{s.getLen():>9,}B]"
            name = s.getPath().getName()
            print(f"  {prefix}{kind}  {name}")
            if s.isDirectory():
                ls_recursive(str(s.getPath()), prefix + "  ")
    except:
        pass

print(f"  {BASE.replace('file://','')}")
ls_recursive(BASE)
captured["final_ls"] = "Final tree: /data/raw, /data/processed/summary_v1, /logs, /output"

# ─── Save ─────────────────────────────────────────────────────────────────────
with open("exp7_results.pkl", "wb") as f:
    pickle.dump({"captured": captured, "N_RECORDS": N_RECORDS, "total_mb": total_mb,
                 "total_files": total_files, "total_dirs": total_dirs,
                 "proc_groups": proc_df.count()}, f)

print("\n✅ Experiment 7 complete. Results saved to exp7_results.pkl")
spark.stop()
