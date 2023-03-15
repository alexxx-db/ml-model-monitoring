# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql.types import *
from scipy.stats import t, poisson
from math import sqrt
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC Get data

# COMMAND ----------

df = spark.table("adult")
# display(df)
print((df.count(), len(df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC Get simple statistics about the data

# COMMAND ----------

stats = df.groupby().agg(
  F.count(F.col("age")),
  F.mean(F.col("age")),
  F.stddev(F.col("age")),
)

display(stats)

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate the t_score for the data and append as a column

# COMMAND ----------

confidence = 0.95

@udf("float")
def t_score(confidence, sample_size):
  dof = sample_size - 1
  alpha_level = (1 - confidence) / 2
  return float(t.ppf(1 - alpha_level, dof))

stats_w_t = df.groupby().agg(
  F.count(F.col("age")).alias("sample_size"),
  F.mean(F.col("age")),
  F.stddev(F.col("age")),
).withColumn(
  "t_score",
  t_score(F.lit(confidence), F.col("sample_size"))
)

display(stats_w_t)

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate the CI for the mean of the data and put in a column

# COMMAND ----------

@udf(ArrayType(FloatType()))
def ci(std, sample_size, t_score, mean):
  tmp = (std / sqrt(sample_size)) * t_score
  return (float(mean - tmp), float(mean + tmp))

stats_w_ci = df.groupby().agg(
  F.count(F.col("age")).alias("sample_size"),
  F.mean(F.col("age")).alias("mean"),
  F.stddev(F.col("age")).alias("std"),
).withColumn(
  "t_score",
  t_score(F.lit(confidence), F.col("sample_size"))
).withColumn(
  "ci",
  ci(F.col("std"), F.col("sample_size"), F.col("t_score"), F.col("mean"))
)

display(stats_w_ci)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Bootstrapping CI
# MAGIC 
# MAGIC 1. Extend the dataset with B columns, for B = # of bootstramp samples (you can try something smaller like 100). The value of column i (1 <= i <= B) is the number of times the tuple appears in the i-th sample. Values are populated with sampling from a Poisson distribution.
# MAGIC 
# MAGIC 2. Create B datasets by projecting on original columns and column {i}, i.e. the original dataset plus the i'th replication column
# MAGIC 
# MAGIC 3. For each dataset, replicate each tuple based on the value in column {i} (and filter out tuples where the replication value is 0). To replicate each row by its replication factor, you may want to look into explode(array_repeat(col, col)). array_repeat can take in a number and make an array with that number of elements in it. explode can take an array and create a row for each value in the array.
# MAGIC 
# MAGIC 4. Compute statistics on each replicated dataset and concatenate them together into a single DataFrame of bootstrap statistics.
# MAGIC 
# MAGIC 5. Calculate CIs from the distribution of the resampled statistics using quantiles - check out the percentile_approx function to get e.g. 1000 quantiles. To get a confidence interval, you can just look at the lower and higher quantiles that cover the width of the interval. Note that there are more sophisticated ways to extract a Confidence interval from this bootstrap distribution, but not necessary for this prototype.

# COMMAND ----------

bootstrap_samples = 100

@udf("int")
def poisson_random(seed):
  return int(poisson.ppf(seed, 1))

dfs = [(
  df.alias(f"dfb{i}").withColumn(
    "r", F.rand()
  ).withColumn(
    "n", poisson_random(F.col("r"))
  ).drop("r").withColumn(
    "n", F.expr('explode(array_repeat(n, int(n)))')
  ).drop("n")) 
  for i in range(bootstrap_samples)]

# COMMAND ----------

b_stats = dfs[0].groupby().agg(
  F.count(F.col("age")),
  F.mean(F.col("age")).alias("mean"),
  F.stddev(F.col("age")),
)

for i in range(1, 100):
  b_stats = b_stats.union(
    dfs[i].groupby().agg(
    F.count(F.col("age")),
    F.mean(F.col("age")).alias("mean"),
    F.stddev(F.col("age")),
  ))

display(b_stats)

# COMMAND ----------

quantiles = b_stats.groupby().agg(
  F.percentile_approx(F.col("mean"), list(np.linspace(0, 1, 1000))).alias("quantiles"),
)

display(quantiles)
