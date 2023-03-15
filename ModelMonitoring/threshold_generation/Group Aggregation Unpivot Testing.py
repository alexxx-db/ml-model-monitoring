# Databricks notebook source
import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

from scipy.stats import t
from timeit import default_timer as timer

# COMMAND ----------

@F.pandas_udf(T.FloatType(), F.PandasUDFType.SCALAR)
def t_score_udf(interval_width: float, sample_size: int) -> float:
    dof = sample_size - 1
    alpha_level = (1 - interval_width) / 2
    return pd.Series(float(t.ppf(1 - alpha_level, dof)))

# COMMAND ----------

def generate_thresholds_stack(baseline_data: pyspark.sql.DataFrame,
                        interval_width: float = 0.99) -> pyspark.sql.DataFrame:
    supported_cols = [
        f.name for f in baseline_data.schema.fields if isinstance(f.dataType, T.NumericType)
    ]

    def gen_expression(col):
        sample_size = F.count(F.col(col))
        mean = F.mean(F.col(col))
        stddev = F.stddev(F.col(col))

        t_score = t_score_udf(F.lit(interval_width), sample_size)
        breadth = t_score * stddev / F.sqrt(sample_size)
        ci_low = mean - breadth
        ci_high = mean + breadth
        return ci_low.alias(f"{col}_ci_low"), ci_high.alias(f"{col}_ci_high")

    ci_df = baseline_data.groupby().agg(
        *[expr for col in supported_cols for expr in gen_expression(col)])

    expr = f"{len(supported_cols)}" + "".join(f", '{c}', {c}_ci_low, {c}_ci_high"
                                              for c in supported_cols)

    return ci_df.select(
        F.lit("avg").alias("metric"),
        F.lit(interval_width).alias("interval_width"),
        F.expr(f"stack({expr}) as (column_name, low, high)"),
    )

# COMMAND ----------

def generate_thresholds_explode(
    baseline_data: pyspark.sql.DataFrame, interval_width: float = 0.99
) -> pyspark.sql.DataFrame:
    supported_cols = [
        f.name
        for f in baseline_data.schema.fields
        if isinstance(f.dataType, T.NumericType)
    ]

    def gen_expression(col):
        sample_size = F.count(F.col(col))
        mean = F.mean(F.col(col))
        stddev = F.stddev(F.col(col))

        t_score = t_score_udf(F.lit(interval_width), sample_size)
        breadth = t_score * stddev / F.sqrt(sample_size)
        ci_low = mean - breadth
        ci_high = mean + breadth
        return ci_low.alias(f"{col}_ci_low"), ci_high.alias(f"{col}_ci_high")

    ci_df = baseline_data.groupby().agg(
        *[expr for col in supported_cols for expr in gen_expression(col)]
    )

    return ci_df.select(
        F.explode(
            F.array(
                *[
                    F.struct(
                        F.lit(c).alias("column_name"),
                        F.col(f"{c}_ci_low").alias("ci_low"),
                        F.col(f"{c}_ci_high").alias("ci_high"),
                    )
                    for c in supported_cols
                ]
            )
        )
    ).select("col.*")

# COMMAND ----------

def time_function(function):
    start = timer()
    function(adult)
    end = timer()
    return end - start

# COMMAND ----------

trial_count = 50
functions = [
    ("Stack", generate_thresholds_stack),
    ("Explode", generate_thresholds_explode)
]

for name, f in functions:
    times = [time_function(f) for _ in range(trial_count)]
    print(f"{name}: {round(np.mean(times), 2)}s avg, {round(np.std(times), 2)}s std")

# COMMAND ----------


