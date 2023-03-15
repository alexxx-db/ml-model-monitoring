# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Union testing - exploring & identifying performant methods for unioning many dataframes in PySpark. Write up found [here](https://docs.google.com/document/d/1RpReNEly_NfEYFkMfd3Vu1z8oi89VFQZy39FxF-WqlY/edit)

# COMMAND ----------

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

def standard_union(dfs):
  return functools.reduce(pyspark.sql.DataFrame.union, dfs)

def rdd_union(dfs):
  first_df = dfs[0]
  return first_df.sql_ctx.createDataFrame(
      first_df.sql_ctx._sc.union([df.rdd for df in dfs]), first_df.schema
  )
  
def binary_union(dfs):
    num_dfs = len(dfs)
    if num_dfs == 1:
        return dfs[0]

    mid_point = num_dfs // 2

    left_partition = binary_union(dfs[:mid_point])
    right_partition = binary_union(dfs[mid_point:])

    return left_partition.union(right_partition)

# COMMAND ----------

def explain_df(function, dfs):
    function(dfs).explain(mode="formatted")
  
def time_union_and_collect(function, dfs):
    start = timer()
    function(dfs).collect()
    end = timer()
    return end - start

# COMMAND ----------

dfs = [spark.table("adult")] * 100

# COMMAND ----------

# MAGIC %md
# MAGIC Compare Physical plans for dfs of different union methods

# COMMAND ----------

explain_df(standard_union, dfs)

# COMMAND ----------

explain_df(rdd_union, dfs)

# COMMAND ----------

explain_df(binary_union, dfs)

# COMMAND ----------

# MAGIC %md
# MAGIC PySpark Unioning

# COMMAND ----------

functions = [
    ("df union", standard_union),
    ("rdd union", rdd_union),
    ("binary union", binary_union)
]
trial_count = 5

for b in range(250, 751, 250):
  dfs = [spark.table("adult")] * b
  print(f"{'-'*24}\nTesting for {b} unions\n{'-'*24}")
  for name, f in functions:
    times = [time_union_and_collect(f, (dfs)) for _ in range(trial_count)]
    print(f"{name}: {round(np.mean(times), 2)}s avg, {round(np.std(times), 2)}s std")

# COMMAND ----------

# MAGIC %md
# MAGIC SQL Unioning

# COMMAND ----------

trial_count = 5

def time_sql_union(count):
    start = timer()
    spark.sql(f"SELECT * FROM adult {' '.join(['union all SELECT * FROM adult'] * (count - 1))}").collect()
    end = timer()
    return end - start

for b in range(200, 501, 100):
  times = [time_sql_union(b) for _ in range(trial_count)]
  print(f"{b} unions: {round(np.mean(times), 2)}s avg, {round(np.std(times), 2)}s std")

# COMMAND ----------

pyspark_union_df = binary_union([spark.table("hive_metastore.automl.creditcard")] * 5)
print((pyspark_union_df.count(), len(pyspark_union_df.columns)))

# COMMAND ----------

sql_union_df = spark.sql(f"SELECT * FROM hive_metastore.automl.creditcard {' '.join(['union all SELECT * FROM hive_metastore.automl.creditcard']* 4)}")
print((sql_union_df.count(), len(sql_union_df.columns)))

# COMMAND ----------

spark.sql(f"SELECT * FROM adult {' '.join(['union all SELECT * FROM adult'] * 199)}").explain("formatted")

# COMMAND ----------


