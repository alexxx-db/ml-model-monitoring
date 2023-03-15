# Databricks notebook source
# MAGIC %md Benchmark sampling from DataFrame

# COMMAND ----------

DEFAULT_SAMPLE_SIZE = 1000

def sample_data_df(df, sample_size=DEFAULT_SAMPLE_SIZE):
  sample_rate = (sample_size * 1.1)/df.count()
  return df.sample(sample_rate).toPandas()

# COMMAND ----------

import pandas as pd
import numpy as np

def check_if_datetime(pd_df):
  for column in pd_df.columns:
    column_vector = pd_df[column].unique()
    result = pd.to_datetime(column_vector, errors="coerce", infer_datetime_format=True)
    errors = result.isna()
    is_datetime = (not errors.all()) and not (errors.sum() > 1)
    #print(f"{column}: {is_datetime}")

# COMMAND ----------

# MAGIC %md Benchmark sampling from RDD

# COMMAND ----------

import pandas as pd

def sample_data_rdd(df, sample_size=DEFAULT_SAMPLE_SIZE):
  sample = df.rdd.takeSample(
    withReplacement=False,
    num=sample_size,
    seed=np.random.randint(1e9),
  )
  return pd.DataFrame(sample)

# COMMAND ----------

# MAGIC %md Benchmark koalas to_datetime

# COMMAND ----------

import databricks.koalas as ks

def check_if_datetime_koalas(df):
  for column in df.columns:
    column_vector = df[column].unique()
    result = ks.to_datetime(column_vector, errors="coerce", infer_datetime_format=True)
    errors = result.isna()
    is_datetime = (not errors.all()) and not (errors.sum() > 1)
    #print(f"{column}: {is_datetime}")

# COMMAND ----------

# MAGIC %md Benchmark pandas UDF

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf

@pandas_udf("boolean")
def check_if_datetime_udf(c: pd.Series) -> bool:
    result = pd.to_datetime(c, errors="coerce", infer_datetime_format=True)
    errors = result.isna()
    is_datetime = (not errors.all()) and not (c[errors].nunique(dropna=False) > 1)
    return is_datetime

# COMMAND ----------

# for column in df.columns:
#   is_datetime = df.select(check_if_datetime_udf(column)).first()[0]
#   print(f"{column}: {is_datetime}")

# COMMAND ----------

import time
from collections import defaultdict
from statistics import mean

import databricks.koalas as ks
from pyspark.sql.types import StringType


num_iters = 5
datasets = ["adult", "hive_metastore.databricks_datasets.amazon_reviews", "artificial", "hive_metastore.databricks_datasets.airlines"]
mem_gb = [0.0021834373474121094, 0.00014882534742355347, 2.980232357978821, 7.91]
all_results = {}
for idx, dataset in enumerate(datasets):
  print(f"benchmarking {dataset}")
  print(f"memory (GB): {mem_gb[idx]}")
  
  if dataset == "artificial":
    df = spark.range(100000000).selectExpr(
      'CONCAT("2020-01-1", id % 10) AS date_str',
      'CONCAT("2020-01-01 00:00:1", id % 10) AS ts_str',
      'IF(id%10==0, NULL, CONCAT("2020-01-1", id % 10)) AS dates_with_nulls',
      'CONCAT("str: ", id % 10000) AS random_str'
    )
  else:
    df = spark.table(dataset)
    
  string_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
  df = df.select(string_cols)
  
  durations = defaultdict(list)
  for i in range(num_iters):
    # DF sample with pd.to_datetime
    tic = time.perf_counter()
    check_if_datetime(sample_data_df(df))
    toc = time.perf_counter()
    durations["df_sample_pd_to_datetime"].append(toc-tic)
    
    # RDD sample with pd.to_datetime
    tic = time.perf_counter()
    check_if_datetime(sample_data_rdd(df))
    toc = time.perf_counter()
    durations["rdd_sample_pd_to_datetime"].append(toc-tic)
    
    # Full dataset with ks.to_datetime
    tic = time.perf_counter()
    check_if_datetime_koalas(ks.DataFrame(df))
    toc = time.perf_counter()
    durations["full_ks_to_datetime"].append(toc-tic)
    
  dataset_result = {}
  for method, results in durations.items():
    dataset_result[method] = f"{mean(results)} seconds"
    
  all_results[dataset] = dataset_result
  print("---------------------------")
  print()
  
print("---------------------------")
print("RESULTS")
print(all_results)

# COMMAND ----------

pd.DataFrame(all_results)

# COMMAND ----------


