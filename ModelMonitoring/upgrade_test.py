# Databricks notebook source
# Install model monitoring client library. Eventually this will be included in MLR
%pip install --upgrade --no-deps --force-reinstall "https://ml-team-public-read.s3.amazonaws.com/wheels/model-monitoring/e7394149-4c48-42a3-94af-11cee8964415/databricks_model_monitoring-0.0.1-py3-none-any.whl"

# COMMAND ----------

dbutils.widgets.text("model_name_prefix", "mldata_adult_census_upgrade_test")
dbutils.widgets.text("monitor_name", "upgrade_test")
dbutils.widgets.text("monitor_db", "quickstart_model_monitor_db")

# Add username as suffix to protect against conflicts
import re
import string
username = dbutils.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
username_sanitized = re.sub(f"[{string.punctuation}{string.whitespace}]", "_", username).lower()
model_name = dbutils.widgets.get("model_name_prefix")+"_"+username_sanitized
monitor_name = dbutils.widgets.get("monitor_name")
monitor_db = dbutils.widgets.get("monitor_db")

# COMMAND ----------

# Global Constant for columns in the income prediction dataset this notebook monitors
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Model Training and Registration

# COMMAND ----------

import mlflow
from mlflow import tracking
from mlflow.tracking.client import MlflowClient
import pandas as pd

from databricks import automl

client = MlflowClient()

# COMMAND ----------

train_pdf = pd.read_csv(
  "/dbfs/databricks-datasets/adult/adult.data",
  names=cols,
  skipinitialspace=True,
)
df_train = spark.createDataFrame(train_pdf)
summary = automl.classify(
  df_train,
  target_col="income",
  timeout_minutes=5
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Registration
# MAGIC We only register the first run out of AutoML

# COMMAND ----------

r = summary.trials[0]
model_version = mlflow.register_model(
    "runs:/{run_id}/model".format(run_id=r.mlflow_run_id),
    model_name
  )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a monitor

# COMMAND ----------

# Model monitoring will store results in delta tables in a a Databricks database, so we make sure the database exists
spark.sql(f"CREATE DATABASE IF NOT EXISTS {monitor_db}")

# COMMAND ----------

import copy
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import types as T
import random
import uuid

import databricks.model_monitoring as mm

# COMMAND ----------

# Define the logging schema which contains all fields to track in the Logging Table
logging_schema = copy.deepcopy(df_train.schema)
logging_schema.add("income_predicted", T.StringType(), False)
logging_schema.add("example_id", T.StringType(), False)

mm_info = mm.create_monitor(
  model_name=model_name,
  monitor_name=monitor_name,
  model_type="classifier",
  database_name=monitor_db,
  logging_schema=logging_schema,
  # This specifies that metrics will be computed based on daily windows
  granularities=["1 day"],
  # This column allows monitoring to compute statistics grouped by (age > 18)
  slicing_exprs=["age > 18"],
  # The column in logging_schema used to store model predictions
  prediction_col="income_predicted",
  # The column in logging_schema used to store ground truth labels
  label_col="income",
  # The id column from logging_schema that uniquely identifies a row to link ground truth labels that arrive later
  id_cols=["example_id"],
)

# COMMAND ----------

# MAGIC %md ## Switch wheels and update_monitor

# COMMAND ----------

# Install model monitoring client library. Eventually this will be included in MLR
%pip install --upgrade --no-deps --force-reinstall "https://ml-team-public-read.s3.amazonaws.com/wheels/model-monitoring/e7394149-4c48-42a3-94af-11cee8964415/databricks_model_monitoring-0.0.2-py3-none-any.whl"

# COMMAND ----------

dbutils.widgets.text("model_name_prefix", "mldata_adult_census_upgrade_test")
dbutils.widgets.text("monitor_name", "upgrade_test")
dbutils.widgets.text("monitor_db", "quickstart_model_monitor_db")

# Add username as suffix to protect against conflicts
import re
import string
username = dbutils.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
username_sanitized = re.sub(f"[{string.punctuation}{string.whitespace}]", "_", username).lower()
model_name = dbutils.widgets.get("model_name_prefix")+"_"+username_sanitized
monitor_name = dbutils.widgets.get("monitor_name")
monitor_db = dbutils.widgets.get("monitor_db")

# COMMAND ----------

import databricks.model_monitoring as mm

mm.upgrade_monitor(model_name=model_name, monitor_name=monitor_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleanup
# MAGIC The following section provides helper methods to clean up the artifacts created in this notebook

# COMMAND ----------

import databricks.model_monitoring as mm
from mlflow import tracking
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

# COMMAND ----------

def cleanup_model(model_name):
  filter_string = "name='{}'".format(model_name)
  versions = client.search_model_versions(filter_string)
  for v in versions:
    if v.current_stage != "Archived":
      client.transition_model_version_stage(
        name=model_name,
        version=v.version,
        stage="Archived",
      )
  client.delete_registered_model(name=model_name)

# COMMAND ----------

mm.delete_monitor(model_name=model_name, monitor_name=monitor_name, purge_tables=True)
cleanup_model(model_name)

# COMMAND ----------


