# Databricks notebook source
# MAGIC %pip install line_profiler

# COMMAND ----------

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

df = spark.table("hive_metastore.automl.creditcard")
col = "V5"
print(f"\n\n creditcard V5: {df.count()}, {len(df.columns)}")

# COMMAND ----------

import inspect
from io import StringIO
from line_profiler import LineProfiler

def profile_function(my_func, *args, **kwargs):
    lp = LineProfiler()
    output_val = lp(my_func)(*args, **kwargs)
    mystdout = StringIO()
    lp.print_stats(stream=mystdout)  # Redirect stdout so we can grab profile output
    lprof_lines = mystdout.getvalue().split("\n")
    profile_start = 1 + next(
        idx for idx, line in enumerate(lprof_lines) if "=====" in line
    )
    lprof_code_lines = lprof_lines[profile_start:-1]
    source_lines = inspect.getsource(my_func).split("\n")[:-1]
    if len(source_lines) != len(lprof_code_lines):
        print("WARNING! Mismatch in source length and returned line profiler estimates")
        print(len(source_lines))
        print(len(lprof_code_lines))
        print("\n".join(lprof_lines))
        print("---- Code ----")
        print(source_lines)
    else:
        print("\n".join(lprof_lines[:profile_start]))
        print(
            "\n".join(
                [
                    "{0} \t {1}".format(l, s)
                    for l, s in zip(lprof_code_lines, source_lines)
                ]
            )
        )
    return output_val

# COMMAND ----------

adult = spark.table("adult")
CONFIDENCE = 0.95
BOOTSTRAP_SAMPLE_COUNT = 100

# COMMAND ----------

@pandas_udf("int", PandasUDFType.SCALAR)
def poisson_random_vec(r):
    return pd.Series(poisson.ppf(r, 1))

def vpsg_bootstrap():
    bootstrap_col = [StructField("bootstrap", IntegerType(), True)]
    vpsg_dfs = []

    for i in range(BOOTSTRAP_SAMPLE_COUNT):
        vpsg_dfs.append(
            adult.alias(f"dfb{i}")
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
    
def vpsgwa_bootstrap():
    bootstrap_col = [StructField("bootstrap", IntegerType(), True)]
    vpsgwa_dfs = []

    for i in range(BOOTSTRAP_SAMPLE_COUNT):
        vpsgwa_dfs.append(
            adult.withColumn("n", F.rand(i))
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
# MAGIC ## Profile VPSG

# COMMAND ----------

profile_function(vpsg_bootstrap)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profile VPSGWA

# COMMAND ----------

profile_function(vpsgwa_bootstrap)

# COMMAND ----------

# MAGIC %scala 
# MAGIC 
# MAGIC val profiler = org.apache.spark.SparkEnv.get.profiler
# MAGIC profiler.warmUp() // This can take a while

# COMMAND ----------

# MAGIC %scala 
# MAGIC 
# MAGIC val run = profiler.run()

# COMMAND ----------


