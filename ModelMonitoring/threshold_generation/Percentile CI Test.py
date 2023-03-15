# Databricks notebook source
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

from collections import defaultdict
from scipy.stats import t, poisson
from typing import List, Mapping, Optional

# COMMAND ----------

@F.pandas_udf(T.FloatType())
def poisson_sample_udf(probability: pd.Series) -> pd.Series:
    return pd.Series(poisson.ppf(probability, 1))


def generate_percentile_threshold(
    baseline_data: pyspark.sql.DataFrame,
    metric_name: str,
    columns: List[str],
    expected_scoring_data_size: int,
    interval_width: float,
) -> pyspark.sql.DataFrame:
    # TODO: think about how to allow input for other percentiles

    # Don't use expected_scoring_data_size adjustment – CI built around the indices, so
    # changing list size will result in inaccurate thresholds
    baseline_data_size = baseline_data.count()
    sample_size = baseline_data_size
    # sample_size = min(expected_scoring_data_size,
    #   baseline_data_size) if expected_scoring_data_size else baseline_data_size

    dof = sample_size - 1
    alpha_level = (1 - interval_width) / 2
    t_score = float(t.ppf(1 - alpha_level, dof))

    def gen_expression(col: str, metric_name: str):
        base_metric = 0.5 * sample_size
        breadth = t_score * math.sqrt(0.25 * sample_size)

        low_index = math.ceil(base_metric - breadth)
        high_index = math.ceil(base_metric + breadth)

        sorted_arr = F.array_sort(F.collect_list(F.col(col)))
        return F.element_at(sorted_arr, low_index).alias(f"{col}_ci_low"), F.element_at(
            sorted_arr, high_index
        ).alias(f"{col}_ci_high")

    ci_df = baseline_data.groupby().agg(
        *[expr for col in columns for expr in gen_expression(col, metric_name)]
    )

    expr = ", ".join(f"'{c}', {c}_ci_low, {c}_ci_high" for c in columns)

    return ci_df.select(
        F.lit(metric_name).alias("metric"),
        F.lit(interval_width).alias("interval_width"),
        F.expr(f"stack({len(columns)}, {expr}) as (column_name, low, high)"),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Test accuracy for adult dataset

# COMMAND ----------

trials = 200
df = spark.table("adult")
thresholds = generate_percentile_threshold(df, "-", ["age"], -1, .95).collect()[0]
low = thresholds.low
high = thresholds.high

percs = []
for i in range(200, 6001, 200):
  success = 0
  for _ in range(trials):
    random_rows = df.rdd.takeSample(False, i)
    if low <= np.median([x.age for x in random_rows]) <= high:
      success += 1
  percs.append(success/trials * 100)
  print(f"For {i} scoring data size, correct threshold {success/trials * 100}%")
  
plt.plot(range(200, 6001, 200), percs)
plt.ylabel('% time sample has fallen in range')
plt.xlabel('Resample count')
plt.ylim(0, 100)
plt.show()

# COMMAND ----------


