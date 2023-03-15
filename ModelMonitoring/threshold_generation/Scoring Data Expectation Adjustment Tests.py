# Databricks notebook source
import functools
import math
import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import seaborn as sns

from scipy.stats import t, poisson

sns.set(rc = {'figure.figsize':(12,6)})

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Verify that replacing the sample size used to compute the confidence intervals can be replaced with the expected scoring data size for the closed form solution. Intuitively, this should make sense as more baseline data would mean metrics that are closer to the true value, and sample size simply widens the interval. This is what we verify in this notebook.
# MAGIC 
# MAGIC We use a normal distribution with mean 8 and standard deviation 5. CI at 95%

# COMMAND ----------

mean = 8
std = 5

# Calculate true confidence intervals for different sample_sizes
confidence_interval_list = []
for i in range(100, 1001, 100):
  ci = 1.96 * std / np.sqrt(i)
  confidence_interval_list.append([mean - ci, i, 'True CI'])
  confidence_interval_list.append([mean + ci, i, 'True CI'])

confidence_intervals = pd.DataFrame(confidence_interval_list, columns=['CI', 'sample_size', 'sample_method'])
ax = sns.pointplot(x="sample_size", y="CI", data=confidence_intervals, capsize=.2, join=False, orient='v', hue="sample_method")

# COMMAND ----------

for i in range(100, 1001, 100):
  data = np.random.normal(8, 5, i)
  ci = 1.96 * np.std(data) / np.sqrt(i)
  mean = np.mean(data)
  confidence_interval_list.append([mean - ci, i, 'True sample size'])
  confidence_interval_list.append([mean + ci, i, 'True sample size'])
  
confidence_intervals = pd.DataFrame(confidence_interval_list, columns=['CI', 'sample_size', 'sample_method'])
ax = sns.pointplot(x="sample_size", y="CI", data=confidence_intervals, capsize=.2, join=False, orient='v', hue="sample_method")

# COMMAND ----------

data = np.random.normal(8, 5, 1000)
mean = np.mean(data)
std = np.std(data)

for i in range(100, 1001, 100):  
  ci = 1.96 * std / np.sqrt(i)
  confidence_interval_list.append([mean - ci, i, 'Adjusted sample size'])
  confidence_interval_list.append([mean + ci, i, 'Adjusted sample size'])
  
confidence_intervals = pd.DataFrame(confidence_interval_list, columns=['CI', 'sample_size', 'sample_method'])
ax = sns.pointplot(x="sample_size", y="CI", data=confidence_intervals, capsize=.2, join=False, orient='v', hue="sample_method")

# COMMAND ----------

@F.pandas_udf("int", F.PandasUDFType.SCALAR)
def poisson_random_vec(r: pd.Series) -> pd.Series:
    return pd.Series(poisson.ppf(r, 1))

def bootstrap(df, b_sample_count, col, interval_width):
    dfs = []

    for i in range(b_sample_count):
        dfs.append(
            df.withColumn("n", F.rand(i))
            .withColumn("n", poisson_random_vec(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .withColumn("bootstrap", F.lit(i))
        )
        
    bootstrap_df = functools.reduce(pyspark.sql.DataFrame.unionByName, dfs)

    metric_df = (
        bootstrap_df.groupby("bootstrap")
        .agg(F.mean(F.col(col)).alias(f"{col}_mean"))
        .drop("bootstrap")
    )
    
    interval_padding = math.floor(((1 - interval_width) / 2) * b_sample_count)
    # Add 1 to indices as SQL is 1-index based
    quantile_index = (
        interval_padding + 1,
        b_sample_count - interval_padding + 1,
    )

    # Generate quantiles for each column-metric pair
    quantiles = metric_df.groupby().agg(
        F.percentile_approx(
            F.col(f"{col}_mean"),
            np.linspace(0, 1, b_sample_count + 1).tolist(),
        ).alias(f"{col}_mean_quantiles")
    )

    return quantiles.select(
            F.element_at(F.col(f"{col}_mean_quantiles"), quantile_index[0]).alias("low"),
            F.element_at(F.col(f"{col}_mean_quantiles"), quantile_index[1]).alias("high"),
        )

# COMMAND ----------

# MAGIC %md
# MAGIC Given the baseline data, find the confidence intervals that are found using the bootstrap method for the corresponding bootstrap sample sizes. 

# COMMAND ----------

# increase bootstrap sample count
# hide true sample size, adjustsed sample size
# compare bootstrap sample size vs true ci

# concept: compare to closed form solution -> how much customers need to pay

df = spark.createDataFrame([float(x) for x in data], T.DoubleType())
for i in range(100, 1001, 100):  
  ci = bootstrap(df, i, 'value', 0.95)
  confidence_interval_list.append([ci.collect()[0][0], i, 'Bootstrap sample size'])
  confidence_interval_list.append([ci.collect()[0][1], i, 'Bootstrap sample size'])
  
confidence_intervals = pd.DataFrame(confidence_interval_list, columns=['CI', 'sample_size', 'sample_method'])
ax = sns.pointplot(x="sample_size", y="CI", data=confidence_intervals, capsize=.2, join=False, orient='v', hue="sample_method")


# COMMAND ----------

# Maybe explore BLB

for i in range(100, 1001, 100):  
  df = spark.createDataFrame([float(x) for x in data if x in sample of i items], T.DoubleType())
  ci = bootstrap(df, i, 'value', 0.95)
  confidence_interval_list.append([ci.collect()[0][0], i, 'Bootstrap sample size'])
  confidence_interval_list.append([ci.collect()[0][1], i, 'Bootstrap sample size'])
  
confidence_intervals = pd.DataFrame(confidence_interval_list, columns=['CI', 'sample_size', 'sample_method'])
ax = sns.pointplot(x="sample_size", y="CI", data=confidence_intervals, capsize=.2, join=False, orient='v', hue="sample_method")

