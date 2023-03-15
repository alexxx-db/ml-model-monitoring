# Databricks notebook source
import functools
from math import sqrt
import numpy as np
import pandas as pd
import pyspark.sql
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from scipy.stats import t, poisson
from timeit import default_timer as timer

# COMMAND ----------

df = spark.table("adult")
CONFIDENCE = 0.95
BOOTSTRAP_SAMPLE_COUNT = 50

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate the CI for the mean of the data and put in a column

# COMMAND ----------

@udf("float")
def t_score(confidence, sample_size):
    dof = sample_size - 1
    alpha_level = (1 - confidence) / 2
    return float(t.ppf(1 - alpha_level, dof))


@udf(ArrayType(FloatType()))
def ci(std, sample_size, t_score, mean):
    tmp = (std / sqrt(sample_size)) * t_score
    return (float(mean - tmp), float(mean + tmp))


stats_w_ci = (
    df.groupby()
    .agg(
        F.count(F.col("age")).alias("sample_size"),
        F.mean(F.col("age")).alias("mean"),
        F.stddev(F.col("age")).alias("std"),
    )
    .withColumn("t_score", t_score(F.lit(CONFIDENCE), F.col("sample_size")))
    .withColumn(
        "ci", ci(F.col("std"), F.col("sample_size"), F.col("t_score"), F.col("mean"))
    )
)

dataFrame.queryExecution.optimizedPlan.stats

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Standard Bootstrapping CI Implementation

# COMMAND ----------

# Standard implementation code: si

@udf("int")
def poisson_random(r):
    return int(poisson.ppf(r, 1))


def si_bootstrap():    
    bootstrap_stat_dfs = [
        (
            df.alias(f"dfb{i}")
            .withColumn("n", F.rand(i))
            .withColumn("n", poisson_random(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .groupby()
            .agg(
                F.count(F.col("age")).alias("count"),
                F.mean(F.col("age")).alias("mean"),
                F.stddev(F.col("age")).alias("stddev"),
            )
        )
        for i in range(BOOTSTRAP_SAMPLE_COUNT)
    ]

    return functools.reduce(pyspark.sql.DataFrame.unionByName, bootstrap_stat_dfs)

# COMMAND ----------

# MAGIC %md
# MAGIC Vectorize with pandas UDF

# COMMAND ----------

# Vectorize with pandas UDF code: vp


@pandas_udf("int", PandasUDFType.SCALAR)
def poisson_random_vec(r):
    return pd.Series(poisson.ppf(r, 1))


def vp_bootstrap():
    vp_stat_dfs = [
        (
            df.alias(f"dfb{i}")
            .withColumn("n", F.rand(i))
            .withColumn("n", poisson_random_vec(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .groupby()
            .agg(
                F.count(F.col("age")).alias("count"),
                F.mean(F.col("age")).alias("mean"),
                F.stddev(F.col("age")).alias("stddev"),
            )
        )
        for i in range(BOOTSTRAP_SAMPLE_COUNT)
    ]

    return functools.reduce(pyspark.sql.DataFrame.unionByName, vp_stat_dfs)

# COMMAND ----------

# MAGIC %md
# MAGIC Single groupby function

# COMMAND ----------

# Single groupby code: sg


@udf("int")
def poisson_random(r):
    return int(poisson.ppf(r, 1))

def sg_bootstrap():
    bootstrap_col = [StructField("bootstrap", IntegerType(), True)]
    sg_dfs = []

    for i in range(BOOTSTRAP_SAMPLE_COUNT):
        sg_dfs.append(
            df.alias(f"dfb{i}")
            .withColumn("n", F.rand(i))
            .withColumn("n", poisson_random(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .withColumn("bootstrap", F.lit(i))
        )
    
    sg_df = functools.reduce(pyspark.sql.DataFrame.unionByName, sg_dfs)

    return (
        sg_df.groupby("bootstrap")
        .agg(
            F.count(F.col("age")).alias("count"),
            F.mean(F.col("age")).alias("mean"),
            F.stddev(F.col("age")).alias("stddev"),
        )
        .drop("bootstrap")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Vectorize with pandas UDF and single groupby

# COMMAND ----------

# Vectorize with pandas UDF and single groupby code: vpsg


@pandas_udf("int", PandasUDFType.SCALAR)
def poisson_random_vec(r):
    return pd.Series(poisson.ppf(r, 1))

def vpsg_bootstrap():
    bootstrap_col = [StructField("bootstrap", IntegerType(), True)]
    vpsg_dfs = []

    for i in range(BOOTSTRAP_SAMPLE_COUNT):
        vpsg_dfs.append(
            df.alias(f"dfb{i}")
            .withColumn("n", F.rand(i))
            .withColumn("n", poisson_random_vec(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .withColumn("bootstrap", F.lit(i))
        )
        
    vpsg_df = functools.reduce(pyspark.sql.DataFrame.unionByName, vpsg_dfs)

    return (
        vpsg_df.groupby("bootstrap")
        .agg(
            F.count(F.col("age")).alias("count"),
            F.mean(F.col("age")).alias("mean"),
            F.stddev(F.col("age")).alias("stddev"),
        )
        .drop("bootstrap")
    )

# COMMAND ----------

# Vectorize with pandas UDF and single groupby code without alias: vpsgwa


@pandas_udf("int", PandasUDFType.SCALAR)
def poisson_random_vec(r):
    return pd.Series(poisson.ppf(r, 1))

def vpsgwa_bootstrap():
    bootstrap_col = [StructField("bootstrap", IntegerType(), True)]
    vpsgwa_dfs = []

    for i in range(BOOTSTRAP_SAMPLE_COUNT):
        vpsgwa_dfs.append(
            df.withColumn("n", F.rand(i))
            .withColumn("n", poisson_random_vec(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .withColumn("bootstrap", F.lit(i))
        )
        
    vpsgwa_df = functools.reduce(pyspark.sql.DataFrame.unionByName, vpsgwa_dfs)

    return (
        vpsgwa_df.groupby("bootstrap")
        .agg(
            F.count(F.col("age")).alias("count"),
            F.mean(F.col("age")).alias("mean"),
            F.stddev(F.col("age")).alias("stddev"),
        )
        .drop("bootstrap")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Minimize column generation code

# COMMAND ----------

# Minimize column generation code: mcgc


@udf("array<int>")
def poisson_random_set(rs):
    return [int(poisson.ppf(seed, 1)) for r in rs]

def mcgc_bootstrap():
    mcgc_df = (
        df.withColumn("n", F.array([F.rand() for _ in range(bootstrap_samples)]))
        .withColumn("n", poisson_random_set(F.col("n")))
    )

    mcgc_stats = spark.createDataFrame(
        [],
        StructType(
            [
                StructField("count", IntegerType(), True),
                StructField("mean", FloatType(), True),
                StructField("stddev", FloatType(), True),
            ]
        ),
    )

    # SQL indexing starts at 1
    for i in range(1, BOOTSTRAP_SAMPLE_COUNT + 1):
        temp_df = mcgc_df.withColumn(
            "n", F.expr(f"explode(array_repeat(n, element_at(n, {i})))")
        ).drop("n")

        mcgc_stats = mcgc_stats.union(
            temp_df.groupby().agg(
                F.count(F.col("age")).alias("count"),
                F.mean(F.col("age")).alias("mean"),
                F.stddev(F.col("age")).alias("stddev"),
            )
        )

    return mcgc_stats

# COMMAND ----------

# MAGIC %md
# MAGIC Time optimizations with and without collection for standard adult dataset

# COMMAND ----------

def time_function(function):
    start = timer()
    function().collect()
    end = timer()
    return end - start

# COMMAND ----------

# MAGIC %md
# MAGIC Time optimizations with and without collection for duplicated adult dataset

# COMMAND ----------

df = spark.table("adult").union(spark.table("adult")).union(spark.table("adult"))

trial_count = 10
functions = [
    ("Pandas UDF + single groupBy", vpsg_bootstrap),
    ("Pandas UDF + single groupBy without alias", vpsgwa_bootstrap)
]
sample_counts = [250, 500, 1000]
for BOOTSTRAP_SAMPLE_COUNT in sample_counts:
    print(f"{'-'*24}\nTesting for {BOOTSTRAP_SAMPLE_COUNT} samples\n{'-'*24}")
    average_times = []
    for name, f in functions:
        times = [time_function(f) for _ in range(trial_count)]
        average_times.append(round(np.mean(times), 2))
        print(f"{name}: {average_times[-1]}s avg, {round(np.std(times), 2)}s std")
    print(f"Speed up: {round(average_times[0] / average_times[-1], 2)}x")
