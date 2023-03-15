# Databricks notebook source
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql
import pyspark.sql.functions as F
import pyspark.sql.types as T
import numpy as np
from timeit import default_timer as timer
from scipy.stats import t, poisson

# COMMAND ----------

@F.pandas_udf(T.FloatType())
def poisson_sample_udf(probability: pd.Series) -> pd.Series:
    return pd.Series(poisson.ppf(probability, 1))


def avg_stddev_bootstrap(baseline):
    dfs = []
    for i in range(BOOTSTRAP_SAMPLE_COUNT):
        dfs.append(
            baseline.withColumn("n", F.rand(i))
            .withColumn("n", poisson_sample_udf(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .withColumn("bootstrap", F.lit(i))
        )

    master_df = _binary_union(dfs)

    metric_df = (
        master_df.groupby("bootstrap")
        .agg(
            F.mean(F.col("age")).alias("age_mean"),
            F.stddev(F.col("age")).alias("age_stddev"),
        )
        .drop("bootstrap")
    )
    interval_padding = math.floor(((1 - CONFIDENCE) / 2) * BOOTSTRAP_SAMPLE_COUNT)
    quantile_index = (
        interval_padding + 1,
        BOOTSTRAP_SAMPLE_COUNT - interval_padding + 1,
    )

    # Generate quantiles for each column-metric pair
    quantiles = metric_df.groupby().agg(
        F.percentile_approx(
            F.col("age_mean"),
            np.linspace(0, 1, BOOTSTRAP_SAMPLE_COUNT + 1).tolist(),
        ).alias("mean_quantiles"),
        F.percentile_approx(
            F.col("age_stddev"),
            np.linspace(0, 1, BOOTSTRAP_SAMPLE_COUNT + 1).tolist(),
        ).alias("std_quantiles"),
    )

    lol = [
        quantiles.select(
            F.element_at(F.col("mean_quantiles"), quantile_index[0]).alias("low"),
            F.element_at(F.col("mean_quantiles"), quantile_index[1]).alias("high"),
        ),
        quantiles.select(
            F.element_at(F.col("std_quantiles"), quantile_index[0]).alias("low"),
            F.element_at(F.col("std_quantiles"), quantile_index[1]).alias("high"),
        ),
    ]
    return _binary_union(lol)


def stddev_bootstrap(baseline):
    dfs = []
    for i in range(BOOTSTRAP_SAMPLE_COUNT):
        dfs.append(
            baseline.withColumn("n", F.rand(i))
            .withColumn("n", poisson_sample_udf(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .withColumn("bootstrap", F.lit(i))
        )

    master_df = _binary_union(dfs)

    return (
        master_df.groupby("bootstrap")
        .agg(F.stddev(F.col("age")).alias("age_stddev"))
        .drop("bootstrap")
    )


def avg_bootstrap(baseline):
    dfs = []
    for i in range(BOOTSTRAP_SAMPLE_COUNT):
        dfs.append(
            baseline.withColumn("n", F.rand(i))
            .withColumn("n", poisson_sample_udf(F.col("n")))
            .withColumn("n", F.expr("explode(array_repeat(n, int(n)))"))
            .drop("n")
            .withColumn("bootstrap", F.lit(i))
        )

    master_df = _binary_union(dfs)

    metric_df = (
        master_df.groupby("bootstrap")
        .agg(F.mean(F.col("age")).alias("age_mean"))
        .drop("bootstrap")
    )
    interval_padding = math.floor(((1 - CONFIDENCE) / 2) * BOOTSTRAP_SAMPLE_COUNT)
    quantile_index = (
        interval_padding + 1,
        BOOTSTRAP_SAMPLE_COUNT - interval_padding + 1,
    )

    # Generate quantiles for each column-metric pair
    quantiles = metric_df.groupby().agg(
        F.percentile_approx(
            F.col("age_mean"),
            np.linspace(0, 1, BOOTSTRAP_SAMPLE_COUNT + 1).tolist(),
        ).alias("age_quantiles")
    )

    return quantiles.select(
        F.element_at(F.col("age_quantiles"), quantile_index[0]).alias("low"),
        F.element_at(F.col("age_quantiles"), quantile_index[1]).alias("high"),
    )


def avg_optim_bootstrap(baseline):
    dfs = []
    for i in range(BOOTSTRAP_SAMPLE_COUNT):
        dfs.append(
            baseline.withColumn("n", F.rand(i))
            .withColumn("n", poisson_sample_udf(F.col("n")))
            .withColumn("bootstrap", F.lit(i))
        )

    master_df = _binary_union(dfs)

    def gen_expression():
        n = F.sum(F.col("n"))
        return (F.sum(F.col("n") * F.col("age")) / n).alias("age_mean")

    metric_df = (master_df.groupby("bootstrap").agg(std_gen_expression())).drop(
        "bootstrap"
    )
    interval_padding = math.floor(((1 - CONFIDENCE) / 2) * BOOTSTRAP_SAMPLE_COUNT)
    quantile_index = (
        interval_padding + 1,
        BOOTSTRAP_SAMPLE_COUNT - interval_padding + 1,
    )

    # Generate quantiles for each column-metric pair
    quantiles = metric_df.groupby().agg(
        F.percentile_approx(
            F.col("age_mean"),
            np.linspace(0, 1, BOOTSTRAP_SAMPLE_COUNT + 1).tolist(),
        ).alias("age_quantiles")
    )

    return quantiles.select(
        F.element_at(F.col("age_quantiles"), quantile_index[0]).alias("low"),
        F.element_at(F.col("age_quantiles"), quantile_index[1]).alias("high"),
    )


def stddev_optim_bootstrap(baseline, col, b, c):
    dfs = []
    for i in range(b):
        dfs.append(
            baseline.withColumn("n", F.rand(i))
            .withColumn("n", poisson_sample_udf(F.col("n")))
            .withColumn("bootstrap", F.lit(i))
        )

    master_df = _binary_union(dfs)

    def std_gen_expression():
        ssq = F.sum(F.col("n") * F.pow(F.col(col), 2))
        sqs = F.sum(F.col(col) * F.col("n"))
        n = F.sum(F.col("n"))
        std = F.sqrt((ssq - (F.pow(sqs, 2) / n)) / (n - 1))
#         return std.alias("stddev")
        return [ssq.alias("ssq"), sqs.alias("sqs"), F.pow(sqs / n, 2).alias("sqs/n squared"), ((ssq / n) - F.pow((sqs / n), 2)).alias(f"ssq/n - sqs/n squared"), F.sqrt((ssq / n) - F.pow(sqs / n, 2)).alias(f"stddev")]
#         return F.sqrt((ssq / n) - F.pow(sqs / n, 2)).alias(f"{col}_stddev")

    metric_df = (master_df.groupby("bootstrap").agg(*std_gen_expression())).drop(
        "bootstrap"
    )
    
    return metric_df
    
    interval_padding = math.floor(((1 - c) / 2) * b)
    quantile_index = (
        interval_padding + 1,
        b - interval_padding + 1,
    )

    # Generate quantiles for each column-metric pair
    quantiles = metric_df.groupby().agg(
        F.percentile_approx(
            F.col(f"{col}_stddev"),
            np.linspace(0, 1, b + 1).tolist(),
        ).alias("std_quantiles")
    )

    return quantiles.select(
        F.element_at(F.col("std_quantiles"), quantile_index[0]).alias("low"),
        F.element_at(F.col("std_quantiles"), quantile_index[1]).alias("high"),
    )


def mixed_bootstrap(baseline):
    return _binary_union([stddev_optim_bootstrap(baseline), avg_bootstrap(baseline)])


def _binary_union(dataframes):
    if len(dataframes) == 1:
        return dataframes[0]

    mid = len(dataframes) // 2

    left = _binary_union(dataframes[:mid])
    right = _binary_union(dataframes[mid:])

    return left.union(right)

# COMMAND ----------

data = {}
data["a"] = np.random.normal(100, 8, 50)

data_df = spark.createDataFrame(pd.DataFrame.from_dict(data))
lol = stddev_optim_bootstrap(data_df, "a", 50, 0.95)

# COMMAND ----------

lol.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time optimizations between optimized standard bootstrap and avg specific

# COMMAND ----------

df = spark.table("adult")
CONFIDENCE = 0.95

def time_function(function):
    start = timer()
    function(df).collect()
    end = timer()
    return end - start

trial_count = 5
functions = [
    ("standard bootstrap", avg_stddev_bootstrap),
    ("mixed bootstrap", mixed_bootstrap)
]

for BOOTSTRAP_SAMPLE_COUNT in range(100, 1001, 100):
    print(f"{'-'*24}\nTesting for {BOOTSTRAP_SAMPLE_COUNT} samples\n{'-'*24}")
    timed = []
    for name, f in functions:
        times = [time_function(f) for _ in range(trial_count)]
        timed.append(round(np.mean(times), 2))
        print(
            f"{name}: {timed[-1]}s avg, {round(np.std(times), 2)}s std"
        )
    print(f"Speed up: {timed[0] / timed[-1]}")

# COMMAND ----------


