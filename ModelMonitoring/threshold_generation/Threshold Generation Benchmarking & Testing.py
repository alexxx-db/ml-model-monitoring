# Databricks notebook source
# Install model monitoring client library. Eventually this will be included in MLR
"""
%pip install --upgrade --no-deps --force-reinstall "https://ml-team-public-read.s3.amazonaws.com/wheels/model-monitoring/e7394149-4c48-42a3-94af-11cee8964415/databricks_model_monitoring-0.0.1-py3-none-any.whl"
"""

%pip install --upgrade --no-deps --force-reinstall "https://databricks-mvn.s3.amazonaws.com/databricks-model-monitoring/databricks-model-monitoring-gdNONWjU.tar.gz?AWSAccessKeyId=AKIAJLF6DMKWZKIYMPAQ&Expires=1966618897&Signature=Ttv7143rGHSAuFegdb6mUXj16H4%3D"

# COMMAND ----------

# Global Constant for columns in the income prediction dataset this notebook monitors
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income']

model_name = "threshold_bat_model"
monitor_name = "threshold_bat_monitor"
monitor_db = "threshold_bat_monitor_db"

# COMMAND ----------

import copy
import databricks.model_monitoring as mm
import mlflow
import numpy as np
import pandas as pd
import pyspark.sql.types as T
import random
import uuid

from databricks import automl
from databricks.model_monitoring import analysis, threshold
from mlflow import tracking
from mlflow.tracking.client import MlflowClient
from pyspark.sql import functions as F
from timeit import default_timer as timer

client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC Empty monitor and model logging table, and relog baseline data (to ensure consistent data being used for benchmarking & performance)

# COMMAND ----------

CLEAR_AND_RELOG_BASELINE_DATA = False

# COMMAND ----------

# Attaches an ID column to allow adding labels to each logged example
def add_id(pdf, seed=0):
  rd = random.Random()
  rd.seed(seed)
  uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128), version=4)
  id_col = [uuid4().hex for _ in range(len(pdf))]
  pdf["example_id"] = id_col
  return pdf

if CLEAR_AND_RELOG_BASELINE_DATA:
  # Get monitor info
  mm_info = mm.get_monitor_info(
    model_name=model_name,
    monitor_name=monitor_name
  )

  # Empty logging table
  logging_table = mm_info.logging_table_name
  spark.sql(f"DELETE FROM {logging_table}")

  prediction_col = mm_info.config.prediction_col
  label_col = mm_info.config.label_col

  # Set up training (baseline) data
  train_pdf = pd.read_csv(
    "/dbfs/databricks-datasets/adult/adult.data",
    names=cols,
    skipinitialspace=True,
  )
  features = [col for col in train_pdf.columns if col not in ['income']]
  train_pdf = add_id(train_pdf, seed=0)

  X_train = train_pdf[features]
  for version in range(1, 4):
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_train)
    train_pdf[prediction_col] = predictions

    # Log baseline data
    baseline_df = spark.createDataFrame(train_pdf)
    mm.log_baseline_data(
      model_name=model_name,
      model_version=str(version),
      monitor_name=monitor_name,
      baseline_data=baseline_df,
      label_col=label_col
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Benchmark threshold generation performance

# COMMAND ----------

times = []

for _ in range(10):
    start = timer()
    threshold.generate_thresholds(
        monitor_name=monitor_name,
        model_name=model_name,
        model_version="1",
        column_metrics={
          "age": ["avg", "stddev"],
          "capital-gain": ["avg", "stddev", "avg_length"],
          "workclass": ["chi_squared_test", "ks_test", "tv_distance"],
          "occupation": ["chi_squared_test", "ks_test", "tv_distance"],
          "capital-loss": ["percent_zeros_delta", "avg_delta"]
        },
        bootstrap_sample_count=50,
    ).collect()
    end = timer()
    times.append(end - start)

print(f"avg time: {round(np.mean(times), 2)} seconds, std: {round(np.std(times), 2)} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC Test threshold generation accuracy

# COMMAND ----------

thresholds = threshold.generate_thresholds(
    monitor_name=monitor_name,
    model_name=model_name,
    model_version="1",
    column_metrics={
      "age": ["avg", "stddev"],
      "capital-gain": ["avg", "stddev", "avg_length"],
      "workclass": ["chi_squared_test", "ks_test", "tv_distance"],
      "occupation": ["chi_squared_test", "ks_test", "tv_distance"],
      "capital-loss": ["percent_zeros_delta", "avg_delta"]
    }
).toPandas()

# COMMAND ----------

test_pdf = pd.read_csv(
  "/dbfs/databricks-datasets/adult/adult.test",
  names=cols,
  skipinitialspace=True,
)

test_pdf["income"].replace(
  {'<=50K.' : '<=50K',
   '>50K.'  : '>50K'},
  inplace=True
)

# Test age average
age_avg_low = thresholds[(thresholds.metric=="avg") & (thresholds.column_name=="age")]["low"].values[0]
age_avg_high= thresholds[(thresholds.metric=="avg") & (thresholds.column_name=="age")]["high"].values[0]
assert age_avg_low <= np.mean(test_pdf["age"].tolist()) <= age_avg_high

# Test age std
age_std_low = thresholds[(thresholds.metric=="stddev") & (thresholds.column_name=="age")]["low"].values[0]
age_std_high= thresholds[(thresholds.metric=="stddev") & (thresholds.column_name=="age")]["high"].values[0]
# assert age_std_low <= np.std(test_pdf["age"].tolist()) <= age_std_high

# Test capital gain average
cg_avg_low = thresholds[(thresholds.metric=="avg") & (thresholds.column_name=="capital-gain")]["low"].values[0]
cg_avg_high= thresholds[(thresholds.metric=="avg") & (thresholds.column_name=="capital-gain")]["high"].values[0]
assert cg_avg_low <= np.mean(test_pdf["capital-gain"].tolist()) <= cg_avg_high

# Test capital gain std
cg_std_low = thresholds[(thresholds.metric=="stddev") & (thresholds.column_name=="capital-gain")]["low"].values[0]
cg_std_high= thresholds[(thresholds.metric=="stddev") & (thresholds.column_name=="capital-gain")]["high"].values[0]
assert cg_std_low <= np.std(test_pdf["capital-gain"].tolist()) <= cg_std_high

# COMMAND ----------


