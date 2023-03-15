# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring Quickstart Notebook
# MAGIC Prerequisites: MLR 10.2+ and Delta Live Tables (optional)
# MAGIC 
# MAGIC See this [folder](https://drive.google.com/drive/folders/1e7sbhf-Hp-gv395XqvIL-4FeYWxUGn20?usp=sharing) for an accompanying **user guide** that introduces the core concepts in Databricks model monitoring, as well as the **API reference** for details on the APIs and tables referenced in this notebook.
# MAGIC 
# MAGIC This quickstart covers a number of common model monitoring CUJs:
# MAGIC - Creating a monitor
# MAGIC - Batch model Inference
# MAGIC - Analyzing model and data quality
# MAGIC - (Bonus) Alerting on model quality
# MAGIC 
# MAGIC Each CUJ can be run independently. 
# MAGIC 
# MAGIC There are also two setup sections. 
# MAGIC - "Library Installation and Parameters" must always be run to prepare the notebook environment
# MAGIC - "Model Training and Registration" only needs to be run once to train a model we will use for monitoring throughout this quickstart.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup: Library Installation and Parameters
# MAGIC 
# MAGIC This notebook is parameterized to create a model monitor named "{monitor_name}", monitoring a model named "{model_name_prefx}_{username}", and to store data in "{monitor_db}". These parameters can be customized from the widgets defined in this section.
# MAGIC 
# MAGIC **Change the model_name and monitor_name parameters to new names to avoid conflicting with existing models and monitors.**
# MAGIC 
# MAGIC **Delete old monitors using the code in the "Cleanup" Section**
# MAGIC 
# MAGIC This notebook also depends on installing the model monitoring client library manually using %pip.

# COMMAND ----------

# Install model monitoring client library. Eventually this will be included in MLR
%pip install --upgrade --no-deps --force-reinstall "https://ml-team-public-read.s3.amazonaws.com/wheels/model-monitoring/e7394149-4c48-42a3-94af-11cee8964415/databricks_model_monitoring-0.0.2-py3-none-any.whl"

# COMMAND ----------

dbutils.widgets.text("model_name_prefix", "mldata_adult_census_quickstart")
dbutils.widgets.text("monitor_name", "quickstart_monitor")
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
# MAGIC 
# MAGIC In this quickstart we train a simple model and then register it as a running example. We then create monitors for this model and its predictions.
# MAGIC 
# MAGIC Below we use [Databricks AutoML](https://docs.databricks.com/applications/machine-learning/automl.html) to train classification models that predict incomes in the adult census dataset.

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
  timeout_minutes=10
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Registration
# MAGIC Model monitoring will monitor all registered versions and stages of a model so we register 3 different versions of the model below, corresponding to different model families that automl explored. In a more realistic deployment, these may correspond to successive versions of a model as the training code and training data are refined.
# MAGIC 
# MAGIC Different versions are transitioned to "Production" and "Staging" stages.

# COMMAND ----------

runs_to_register = [
  [t for t in summary.trials if "logistic" in t.model_description.lower()][0],
  [t for t in summary.trials if "xgb" in t.model_description.lower()][0],
  [t for t in summary.trials if "decisiontree" in t.model_description.lower()][0]
]
stages = ["Production", "Staging", "None"]
for r, stage in zip(runs_to_register, stages):
  model_version = mlflow.register_model(
    "runs:/{run_id}/model".format(run_id=r.mlflow_run_id),
    model_name
  )
  if stage != "None":
    client.transition_model_version_stage(model_name, model_version.version, stage)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a monitor
# MAGIC Now that we have trained and registered models, we can create a monitor for these monitors. 
# MAGIC 
# MAGIC This monitor will track daily statistics on the prediction input data and the quality of income predictions. The monitor is also configured to "slice" data by computing statistics grouped by subsets of data with `"age > 18 = True"` or `"age > 18 = False"`, in addition to statistics computed globally.
# MAGIC 
# MAGIC Note that a model can have one or more monitors attached to it.

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

# Model monitoring will store results in delta tables in a a Databricks database, so we make sure the database exists
spark.sql(f"CREATE DATABASE IF NOT EXISTS {monitor_db}")

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

monitor_ui_link = f"""Monitor Created: <a href="#mlflow/models/{mm_info.model_name}/monitoring">{monitor_name}</a>"""
displayHTML(monitor_ui_link)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Baselines
# MAGIC By optionally adding baselines, model monitoring can compute drift statistics between current
# MAGIC scoring data and the baseline distributions. Thus below we register the training data as a baseline for monitoring. 
# MAGIC 
# MAGIC We attach an `example_id` column to the data as `example_id` is a required field specified in `create_monitor()`. 
# MAGIC 
# MAGIC Model monitoring expects the data to be in a Spark Dataframe, but in this notebook we make liberal use pandas to simplify ad-hoc data manipulations before converting to a Spark Dataframe.

# COMMAND ----------

# Attaches an ID column to allow adding labels to each logged example
def add_id(pdf, seed=0):
  rd = random.Random()
  rd.seed(seed)
  uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128), version=4)
  id_col = [uuid4().hex for _ in range(len(pdf))]
  pdf["example_id"] = id_col
  return pdf

# COMMAND ----------

prediction_col = mm_info.config.prediction_col
label_col = mm_info.config.label_col

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
    baseline_data=baseline_df
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Model Inference
# MAGIC After a model owner has configured monitoring, model consumers can use the model to make predictions and log the results.
# MAGIC 
# MAGIC Below we simulate production batch inference by attaching an event time and example id to each model prediction.
# MAGIC 
# MAGIC We also simulate a data quality issue by zeroing out and dropping key features for a segment of model prediction inputs.

# COMMAND ----------

# Simulate production model scoring data by adding date and example-id columns to our example dataset
def add_date_and_id(pdf, seed=0):
  # spans 15 days in one batch
  dt_col = [
    np.datetime64("2021-01-01") + 
    np.timedelta64(int(idx/1000), 'D') for idx in range(len(pdf))
  ]
  pdf["dt"] = dt_col
  
  rd = random.Random()
  rd.seed(seed)
  uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128), version=4)
  id_col = [uuid4().hex for _ in range(len(pdf))]
  pdf["example_id"] = id_col
  return pdf

# COMMAND ----------

test_pdf = pd.read_csv(
  "/dbfs/databricks-datasets/adult/adult.test",
  names=cols,
  skipinitialspace=True,
)
# We need to clean this dataset to make its labels consistent with training data
test_pdf["income"].replace(
  {'<=50K.' : '<=50K',
   '>50K.'  : '>50K'},
  inplace=True
)
test_pdf = add_date_and_id(test_pdf, seed=1)

# COMMAND ----------

# Simulate a data quality issue by zeroing out a feature on recent data before inferencee
test_pdf.loc[test_pdf["dt"] >= "2021-01-16", "capital-gain"] = 0.0
test_pdf.loc[test_pdf["dt"] >= "2021-01-16", "marital-status"] = None

# COMMAND ----------

label_col = mm_info.config.label_col

# Simulate production traffic for all 3 registered model versions separately
for version in range(1, 4):
  model_uri = f"models:/{model_name}/{version}"
  model = mlflow.pyfunc.load_model(model_uri)
  
  # Sample the scoring data to simulate different workloads for different model versions
  sample_fraction = np.random.uniform(0.6, 0.8)
  test_pdf_sampled = test_pdf.sample(frac=sample_fraction)
  X_test = test_pdf_sampled[features]
  
  # Attach the model predictions to the input data
  predictions = model.predict(X_test)
  test_pdf_sampled[prediction_col] = predictions
  
  # Log scoring data
  scoring_df = spark.createDataFrame(test_pdf_sampled)
  mm.log_scored_data(
    model_name=model_name,
    model_version=str(version),
    scored_data=scoring_df,
    monitor_name=monitor_name,
    timestamp_col="dt"
  )
  
  scoring_labels = scoring_df.select(label_col, "example_id")
  # Log labels for scoring data
  mm.log_labels(
    model_name=model_name,
    labeled_data=scoring_labels,
    monitor_name=monitor_name,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Results
# MAGIC Once the model predictions and input data have been logged, the model monitoring analysis job can compute metrics on the data.
# MAGIC 
# MAGIC The generated multi-task job will run a DLT pipeline that updates the metrics in the analysis tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Triggering the Job
# MAGIC For the purpose of this demo, we trigger the job manually through the `run_analysis` API. However, you're free to inspect the job and unpause it's schedule to automatically refresh the results.

# COMMAND ----------

analysis_job_link = f"""Inspect the <a href="#job/{mm_info.analysis_job_id}">analysis job.</a>"""
displayHTML(analysis_job_link)

mm.run_analysis(
    model_name=model_name,
    monitor_name=monitor_name,
    await_completion=True
)

# COMMAND ----------

# MAGIC %md #### Manual update
# MAGIC 
# MAGIC As a secondary option, if Delta Live Tables (DLT) is not available, one can manually execute the logic to update the tables directly in this notebook.
# MAGIC This workaround is commented below for your reference if DLT is not available to you at this time.
# MAGIC 
# MAGIC NOTE: The code in the helper function below is not an officially supported part of the public API should not be relied upon for production workloads.

# COMMAND ----------

from databricks.model_monitoring import analysis

def run_analysis_manually(model_name, monitor_name):
  info = mm.get_monitor_info(model_name, monitor_name)
  runner = analysis.get_runner(
    model_name=model_name, 
    monitor_name=monitor_name, 
    # This parameter disables certain advanced metrics when true, for compatibility with older runtimes
    dbr_91_compatible=False
  )
  logging_df = spark.table(info.logging_table_name)
  aggregate_df = runner.gen_aggregate_metrics(logging_df)
  derived_df = runner.gen_derived_metrics(aggregate_df)
  drift_df = runner.gen_drift_metrics(derived_df)

  aggregate_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(info.aggregates_base_table_name)
  derived_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(info.analysis_metrics_table_name)
  drift_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(info.drift_metrics_table_name)
  return info.analysis_metrics_table_name, info.drift_metrics_table_name

# Uncomment and run to update the analysis tables manually
# run_analysis_manually(model_name, monitor_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Examine Dashboard
# MAGIC After the analysis job runs you can view the results in the generated notebook dashboard. 
# MAGIC 
# MAGIC Run the dashboard notebook to generate monitoring charts, and then switch to the Dashboard available in the "View" menu to explore results.
# MAGIC One can add additional charts to the generated dashboard as well by adding additional cells that use the provided helper functions.

# COMMAND ----------

dashboard_link = f"""Run and examine generated <a href="#workspace/{mm_info.dashboard_notebook_path}">analysis dashboard</a>."""
displayHTML(dashboard_link)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Manually Query Analysis Tables
# MAGIC For more detailed querying, one can examine the computed analysis tables directly. These tables are easy to query using SQL as long as one makes sure to filter based on any
# MAGIC - model versions
# MAGIC - model stages
# MAGIC - slice key
# MAGIC - slice values
# MAGIC - column names
# MAGIC - window time ranges
# MAGIC 
# MAGIC that one would like to focus on. See the table schemas provided in the reference for more details.
# MAGIC Below we provide some example queries:

# COMMAND ----------

df = spark.sql("""
SELECT window.start, column_name, count, num_nulls, distinct_count
FROM {analysis_table}
WHERE model_version = 1
  AND model_stage = "*"
  AND slice_key IS NULL 
  AND column_name = "income_predicted"
ORDER BY window.start DESC
""".format(
  analysis_table=mm_info.analysis_metrics_table_name
))
display(df)

# COMMAND ----------

df = spark.sql("""
SELECT window.start, model_version, count, accuracy_score
FROM {analysis_table}
WHERE model_version = 1
  AND model_stage = "*"
  AND slice_key IS NULL 
  AND column_name = ":table"
ORDER BY window.start DESC
""".format(
  analysis_table=mm_info.analysis_metrics_table_name
))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alerting
# MAGIC One can set up alerts on the metrics in the analysis and drift tables using [Databricks SQL alerts](https://docs.databricks.com/sql/user/alerts/index.html). Below we provide an example query one can use to alert when the accuracy of the latest production model version falls below a certain threshold. To set up the alert navigate to the SQL alerts page and create an alert based on the query provided below.
# MAGIC 
# MAGIC If one sets up an alert set up based on a threshold of 0.82 then one can see that it would have triggered for the most recent two days of data. This alert would have detected the presence of the data quality issue we introduced eariler!

# COMMAND ----------

alertable_query = """SELECT window.start, model_version, accuracy_score
FROM {analysis_table}
WHERE model_stage = "Production"
  AND slice_key IS NULL 
  AND column_name = ":table"
ORDER BY window.start DESC, model_version DESC
""".format(analysis_table=mm_info.analysis_metrics_table_name)
df = spark.sql(alertable_query)
display(df)

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

# Uncomment to delete the monitor and its artifacts

# mm.delete_monitor(model_name=model_name, monitor_name=monitor_name, purge_tables=True)

# cleanup_model(model_name)
