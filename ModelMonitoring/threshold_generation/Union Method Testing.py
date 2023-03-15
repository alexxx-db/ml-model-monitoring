# Databricks notebook source
import functools
import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

from math import sqrt
from scipy.stats import t, poisson
from timeit import default_timer as timer

# COMMAND ----------

@F.pandas_udf("float")
def poisson_random_vec(r : pd.Series) -> pd.Series:
    return pd.Series(poisson.ppf(r, 1))

  
def bootstrap_union(baseline, column_name, bootstrap_sample_count):
    bootstrap_dfs = []

    for i in range(bootstrap_sample_count):
        bootstrap_dfs.append(
            baseline.withColumn("n", F.rand(i))
            .withColumn("n", poisson_random_vec(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .withColumn("bootstrap", F.lit(i))
        )

    bootstrap_df = functools.reduce(pyspark.sql.DataFrame.union, bootstrap_dfs)

    return (
        bootstrap_df.groupby("bootstrap")
        .agg(
            F.count(F.col(column_name)).alias("count"),
            F.mean(F.col(column_name)).alias("mean"),
            F.stddev(F.col(column_name)).alias("stddev"),
        )
        .drop("bootstrap")
    )


def bootstrap_rdd(baseline, column_name, bootstrap_sample_count):
    def uniondfs(dfs):
        firstDf = dfs[0]
        return firstDf.sql_ctx.createDataFrame(
            firstDf.sql_ctx._sc.union([df.rdd for df in dfs]), firstDf.schema
        )

    bootstrap_dfs = []

    for i in range(bootstrap_sample_count):
        bootstrap_dfs.append(
            baseline.withColumn("n", F.rand(i))
            .withColumn("n", poisson_random_vec(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .withColumn("bootstrap", F.lit(i))
        )

    bootstrap_df = uniondfs(bootstrap_dfs)

    return (
        bootstrap_df.groupby("bootstrap")
        .agg(
            F.count(F.col(column_name)).alias("count"),
            F.mean(F.col(column_name)).alias("mean"),
            F.stddev(F.col(column_name)).alias("stddev"),
        )
        .drop("bootstrap")
    )
    
def binary_union(dfs):
    num_dfs = len(dfs)
    if num_dfs == 1:
        return dfs[0]

    mid_point = num_dfs // 2

    left_partition = binary_union(dfs[:mid_point])
    right_partition = binary_union(dfs[mid_point:])

    return left_partition.union(right_partition)

def bootstrap_binary_union(baseline, column_name, bootstrap_sample_count):
  bootstrap_dfs = []

  for i in range(bootstrap_sample_count):
      bootstrap_dfs.append(
          baseline.withColumn("n", F.rand(i))
          .withColumn("n", poisson_random_vec(F.col("n")))
          .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
          .drop("n")
          .withColumn("bootstrap", F.lit(i))
      )

  bootstrap_df = binary_union(bootstrap_dfs)

  return (
      bootstrap_df.groupby("bootstrap")
      .agg(
          F.count(F.col(column_name)).alias("count"),
          F.mean(F.col(column_name)).alias("mean"),
          F.stddev(F.col(column_name)).alias("stddev"),
      )
      .drop("bootstrap")
  )


# COMMAND ----------

def time_function(function, df_col):
    start = timer()
    function(*df_col).collect()
    end = timer()
    return end - start

# COMMAND ----------

trial_count = 5
functions = [
#     ("df union", bootstrap_union),
#     ("rdd union", bootstrap_rdd),
    ("binary union", bootstrap_binary_union)
]

# df = spark.table("adult").union(spark.table("adult"))
# col = "age"
# print(f"double adult age: {df.count()}, {len(df.columns)}")

# for b in range(250, 1001, 250):
#     print(f"{'-'*24}\nTesting for {b} samples\n{'-'*24}")
#     average_times = []
#     for name, f in functions:
#         times = [time_function(f, (df, col, b)) for _ in range(trial_count)]
#         average_times.append(round(np.mean(times), 2))
#         print(f"{name}: {average_times[-1]}s avg, {round(np.std(times), 2)}s std")

df = spark.table("hive_metastore.automl.creditcard")
col = "V5"
print(f"\n\n creditcard V5: {df.count()}, {len(df.columns)}")

# for b in range(250, 1001, 250):
for b in [1000]:
    print(f"{'-'*24}\nTesting for {b} samples\n{'-'*24}")
    average_times = []
    for name, f in functions:
        times = [time_function(f, (df, col, b)) for _ in range(trial_count)]
        average_times.append(round(np.mean(times), 2))
        print(f"{name}: {average_times[-1]}s avg, {round(np.std(times), 2)}s std")
        
# df = spark.table("hive_metastore.tpch_sf1_delta.orders")
# col = "o_orderkey"
# print(f"\n\n tpch_sf1_delta.orders orderkey: {df.count()}, {len(df.columns)}")

# for b in range(250, 1001, 250):
#     print(f"{'-'*24}\nTesting for {b} samples\n{'-'*24}")
#     average_times = []
#     for name, f in functions:
#         times = [time_function(f, (df, col, b)) for _ in range(trial_count)]
#         average_times.append(round(np.mean(times), 2))
#         print(f"{name}: {average_times[-1]}s avg, {round(np.std(times), 2)}s std")
