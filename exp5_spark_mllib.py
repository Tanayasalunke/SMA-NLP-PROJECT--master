"""
Experiment 5 - SMLBDA
Classification and Clustering using Apache Spark MLlib on Big Data
Student: Tanaya Salunke | Roll: 23101A0074 | INFT-A
"""

import os
import sys
import subprocess

# ─── Set JAVA_HOME ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Try known JDK paths
for _jdk_candidate in [
    os.path.join(SCRIPT_DIR, "smlbda_experiment_8_spark", "jdk-11.0.22+7"),
    "/usr/local/opt/openjdk@11",
    "/usr/local/opt/openjdk@17",
    "/opt/homebrew/opt/openjdk@11",
    "/opt/homebrew/opt/openjdk@17",
]:
    if os.path.exists(os.path.join(_jdk_candidate, "bin", "java")):
        JAVA_HOME = _jdk_candidate
        break
else:
    JAVA_HOME = None
if JAVA_HOME and os.path.exists(JAVA_HOME):
    os.environ["JAVA_HOME"] = JAVA_HOME
    os.environ["PATH"] = os.path.join(JAVA_HOME, "bin") + os.pathsep + os.environ.get("PATH", "")

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, stddev
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, NaiveBayes
)
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, ClusteringEvaluator
)
import random
import math

# ─── Capture output ────────────────────────────────────────────────────────────
output_sections = {}

def header(label):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"{sep}\n")

# ─── 1. SparkSession ───────────────────────────────────────────────────────────
header("1. Initialising SparkSession")
spark = SparkSession.builder \
    .appName("SMLBDA_Exp5_MLlib") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"Spark version : {spark.version}")
print(f"App name      : {spark.sparkContext.appName}")
print("SparkSession created successfully.")
output_sections["spark_init"] = f"Spark Version: {spark.version}\nApp: {spark.sparkContext.appName}\nSparkSession initialized successfully."

# ─── 2. Synthetic Big Dataset ─────────────────────────────────────────────────
header("2. Generating Synthetic Dataset (5000 rows)")
random.seed(42)
N = 5000
raw_data = []
for i in range(N):
    f1 = round(random.gauss(0, 1), 4)
    f2 = round(random.gauss(0, 1), 4)
    f3 = round(random.gauss(0, 1), 4)
    f4 = round(random.gauss(0, 1), 4)
    # label: 0,1,2 based on simple rule
    if f1 + f2 > 0.5:
        label = "A"
    elif f3 - f4 > 0.3:
        label = "B"
    else:
        label = "C"
    raw_data.append((f1, f2, f3, f4, label))

schema = ["feature1", "feature2", "feature3", "feature4", "category"]
df = spark.createDataFrame(raw_data, schema)
df.printSchema()
df.show(10)
print(f"Total records : {df.count()}")
print("Class distribution:")
df.groupBy("category").agg(count("*").alias("count")).orderBy("category").show()

dist_rows = df.groupBy("category").count().collect()
dist_str = "\n".join([f"  {r['category']}: {r['count']} records" for r in dist_rows])
output_sections["dataset"] = (
    f"Generated {N} synthetic records with 4 numeric features and 3 classes (A, B, C).\n"
    f"Schema: feature1 DOUBLE, feature2 DOUBLE, feature3 DOUBLE, feature4 DOUBLE, category STRING\n"
    f"Class Distribution:\n{dist_str}"
)

# ─── 3. Feature Engineering ───────────────────────────────────────────────────
header("3. Feature Engineering")
indexer = StringIndexer(inputCol="category", outputCol="label")
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3", "feature4"],
    outputCol="raw_features"
)
scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=True, withStd=True)

pipeline_prep = Pipeline(stages=[indexer, assembler, scaler])
prepared = pipeline_prep.fit(df).transform(df)
print("Sample prepared data (features + label):")
prepared.select("features", "label", "category").show(5, truncate=80)

train_df, test_df = prepared.randomSplit([0.8, 0.2], seed=42)
print(f"Train size: {train_df.count()} | Test size: {test_df.count()}")
output_sections["features"] = (
    "Feature Engineering Pipeline:\n"
    "  1. StringIndexer: Converts category (A/B/C) -> numeric label (0/1/2)\n"
    "  2. VectorAssembler: Combines feature1..feature4 into a dense vector\n"
    "  3. StandardScaler: Zero-mean, unit-variance normalisation\n\n"
    f"Train size: {train_df.count()}  |  Test size: {test_df.count()}"
)

# ─── 4. CLASSIFICATION ────────────────────────────────────────────────────────
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction"
)
clf_results = {}

def evaluate_classifier(name, model, test):
    preds = model.transform(test)
    acc  = evaluator.evaluate(preds, {evaluator.metricName: "accuracy"})
    prec = evaluator.evaluate(preds, {evaluator.metricName: "weightedPrecision"})
    rec  = evaluator.evaluate(preds, {evaluator.metricName: "weightedRecall"})
    f1   = evaluator.evaluate(preds, {evaluator.metricName: "f1"})
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    clf_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    return preds

# 4a. Logistic Regression
header("4a. Logistic Regression")
lr = LogisticRegression(featuresCol="features", labelCol="label",
                        maxIter=100, regParam=0.01, elasticNetParam=0.0,
                        family="multinomial")
lr_model = lr.fit(train_df)
print("Logistic Regression Results:")
lr_preds = evaluate_classifier("Logistic Regression", lr_model, test_df)
lr_preds.select("category", "label", "prediction").show(10)

# 4b. Decision Tree
header("4b. Decision Tree Classifier")
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                             maxDepth=5, impurity="gini")
dt_model = dt.fit(train_df)
print("Decision Tree Results:")
dt_preds = evaluate_classifier("Decision Tree", dt_model, test_df)
dt_preds.select("category", "label", "prediction").show(10)

# 4c. Random Forest
header("4c. Random Forest Classifier")
rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                             numTrees=50, maxDepth=5, seed=42)
rf_model = rf.fit(train_df)
print("Random Forest Results:")
rf_preds = evaluate_classifier("Random Forest", rf_model, test_df)
rf_preds.select("category", "label", "prediction").show(10)
print("Feature Importances (Random Forest):")
for i, imp in enumerate(rf_model.featureImportances.toArray()):
    print(f"  feature{i+1}: {imp:.4f}")

# 4d. Naive Bayes
header("4d. Naive Bayes Classifier")
# Naive Bayes needs non-negative features; use MinMaxScaler on top of features
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
mm_scaler = MinMaxScaler(inputCol="features", outputCol="features_nb")
train_nb = mm_scaler.fit(train_df).transform(train_df)
test_nb  = mm_scaler.fit(test_df).transform(test_df)

nb = NaiveBayes(featuresCol="features_nb", labelCol="label", modelType="multinomial")
nb_model = nb.fit(train_nb)
print("Naive Bayes Results:")
nb_preds = evaluate_classifier("Naive Bayes", nb_model, test_nb)
nb_preds.select("category", "label", "prediction").show(10)

# ─── 5. Classification Comparison Summary ─────────────────────────────────────
header("5. Classification Results Summary")
print(f"{'Algorithm':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)
for algo, metrics in clf_results.items():
    print(f"{algo:<22} {metrics['Accuracy']:>10.4f} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} {metrics['F1']:>10.4f}")

clf_summary = "Classification Comparison:\n"
clf_summary += f"{'Algorithm':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n"
clf_summary += "-" * 65 + "\n"
for algo, m in clf_results.items():
    clf_summary += f"{algo:<22} {m['Accuracy']:>10.4f} {m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1']:>10.4f}\n"
best_clf = max(clf_results, key=lambda k: clf_results[k]["F1"])
clf_summary += f"\nBest Classifier by F1-Score: {best_clf} ({clf_results[best_clf]['F1']:.4f})"
output_sections["classification"] = clf_summary
print(f"\nBest Classifier: {best_clf} (F1={clf_results[best_clf]['F1']:.4f})")

# ─── 6. CLUSTERING ────────────────────────────────────────────────────────────
clustering_df = prepared.select("features")
# Use cosine distance (works with all clustering algorithms in Spark 4.x)
clust_evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette",
                                      distanceMeasure="cosine")
clust_results = {}

def evaluate_cluster(name, model, data):
    preds = model.transform(data)
    try:
        sil = clust_evaluator.evaluate(preds)
    except Exception as e:
        print(f"  [Silhouette fallback] {e}")
        # Fallback: squared euclidean
        fallback_eval = ClusteringEvaluator(featuresCol="features", metricName="silhouette",
                                            distanceMeasure="squaredEuclidean")
        try:
            sil = fallback_eval.evaluate(preds)
        except Exception:
            sil = 0.0
    print(f"  Silhouette Score: {sil:.4f}")
    print("  Cluster Sizes:")
    preds.groupBy("prediction").count().orderBy("prediction").show()
    clust_results[name] = sil
    return preds

# 6a. K-Means
header("6a. K-Means Clustering (k=3)")
km = KMeans(k=3, seed=42, featuresCol="features", predictionCol="prediction", maxIter=20)
km_model = km.fit(clustering_df)
print("K-Means Cluster Centers:")
for i, c in enumerate(km_model.clusterCenters()):
    print(f"  Center {i}: {[round(x, 3) for x in c.tolist()]}")
km_preds = evaluate_cluster("K-Means (k=3)", km_model, clustering_df)

# K-Means Elbow
header("6b. K-Means Elbow Method (k=2..6)")
wssse_list = []
for k in range(2, 7):
    m = KMeans(k=k, seed=42, featuresCol="features", maxIter=20).fit(clustering_df)
    wssse = m.summary.trainingCost
    wssse_list.append((k, round(wssse, 2)))
    print(f"  k={k}  WSSSE={wssse:.2f}")

elbow_str = "\n".join([f"  k={k}: WSSSE={w}" for k, w in wssse_list])
output_sections["elbow"] = f"Elbow Method Results:\n{elbow_str}"

# 6c. Bisecting K-Means
header("6c. Bisecting K-Means (k=3)")
bkm = BisectingKMeans(k=3, seed=42, featuresCol="features", maxIter=20)
bkm_model = bkm.fit(clustering_df)
bkm_preds = evaluate_cluster("Bisecting K-Means (k=3)", bkm_model, clustering_df)

# 6d. Gaussian Mixture Model
header("6d. Gaussian Mixture Model (k=3)")
gmm = GaussianMixture(k=3, seed=42, featuresCol="features", maxIter=50, tol=0.01)
gmm_model = gmm.fit(clustering_df)
print("GMM Gaussians:")
for i, w in enumerate(gmm_model.weights):
    print(f"  Component {i}: weight={w:.4f}")
gmm_preds = evaluate_cluster("GMM (k=3)", gmm_model, clustering_df)

# ─── 7. Clustering Summary ─────────────────────────────────────────────────────
header("7. Clustering Results Summary")
print(f"{'Algorithm':<25} {'Silhouette Score':>18}")
print("-" * 45)
for algo, sil in clust_results.items():
    print(f"{algo:<25} {sil:>18.4f}")

clust_summary = "Clustering Silhouette Scores:\n"
clust_summary += f"{'Algorithm':<25} {'Silhouette Score':>18}\n"
clust_summary += "-" * 45 + "\n"
for algo, sil in clust_results.items():
    clust_summary += f"{algo:<25} {sil:>18.4f}\n"
best_clust = max(clust_results, key=lambda k: clust_results[k])
clust_summary += f"\nBest Clustering by Silhouette: {best_clust} ({clust_results[best_clust]:.4f})"
output_sections["clustering"] = clust_summary
print(f"\nBest Clustering: {best_clust}")

# ─── 8. Persist results ────────────────────────────────────────────────────────
import pickle
with open("exp5_results.pkl", "wb") as f:
    pickle.dump({
        "clf_results": clf_results,
        "clust_results": clust_results,
        "wssse_list": wssse_list,
        "output_sections": output_sections,
        "best_clf": best_clf,
        "best_clust": best_clust,
    }, f)

print("\n✅ All experiments completed. Results saved to exp5_results.pkl")
spark.stop()
