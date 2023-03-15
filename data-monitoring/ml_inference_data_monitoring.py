# Databricks notebook source
# MAGIC %md
# MAGIC # Data Monitoring Quickstart for ML Inference-like table
# MAGIC 
# MAGIC **System requirements:**
# MAGIC - ML runtime [12.0+ ](https://docs.databricks.com/release-notes/runtime/12.0ml.html)
# MAGIC - [Unity-Catalog enabled workspace](https://docs.databricks.com/data-governance/unity-catalog/enable-workspaces.html)
# MAGIC - Disable **Customer-Managed Key(s)** for encryption [AWS](https://docs.databricks.com/security/keys/customer-managed-keys-managed-services-aws.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/security/keys/customer-managed-key-managed-services-azure) | [GCP]()
# MAGIC 
# MAGIC [Link](https://drive.google.com/drive/u/0/folders/1oXuP-VleXmq0fTE4YovavboAVC7L-DF5) to Google Drive containing:
# MAGIC - User guide on core concepts
# MAGIC - API reference for API details and guidelines 
# MAGIC 
# MAGIC 
# MAGIC In this notebook, we'll train and deploy an ML _regression_ model and monitor its corresponding _(batch)_ **Inference Table**.

# COMMAND ----------

# DBTITLE 1,Install data monitoring wheel
# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_data_monitoring-0.1.0-py3-none-any.whl"

# COMMAND ----------

import databricks.data_monitoring as dm
from databricks.data_monitoring import analysis

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
username_prefix = username.split("@")[0].replace(".","_")

dbutils.widgets.dropdown("problem_type", "regression", ["classification", "regression"],"ML Problem Type (Required)")
dbutils.widgets.text("model_name", f"{username_prefix}_airbnb_pricer", "MLflow Model Name (OPTIONAL)")
dbutils.widgets.text("table_name",f"{username_prefix}_airbnb_pricer_ml_inference", "Table to Monitor")
dbutils.widgets.text("monitor_db", "default", "Output Database/Schema to use (OPTIONAL)")
dbutils.widgets.text("monitor_catalog", "dm_bugbash", "Unity Catalog to use (Required)")

# COMMAND ----------

# MAGIC %md ## Helper methods
# MAGIC 
# MAGIC The function(s) are for cleanup, if the notebook has been run multiple times. You are not expected to use these functions in a normal setup.

# COMMAND ----------

from mlflow.client import MlflowClient

def cleanup_registered_model(registry_model_name):
  """
  Utilty function to delete a registered model in MLflow model registry.
  To delete a model in the model registry, all model versions must first be archived.
  This function 
  (i) first archives all versions of a model in the registry
  (ii) then deletes the model 
  
  :param registry_model_name: (str) Name of model in MLflow Model Registry
  """

  filter_string = f'name="{registry_model_name}"'
  model_exist = client.search_registered_models(filter_string=filter_string)

  if model_exist:
    model_versions = client.search_model_versions(filter_string=filter_string)
    print(f"Deleting model named {registry_model_name}...")
    if len(model_versions) > 0:
      print(f"Purging {len(model_versions)} versions...")
      # Move any versions of the model to Archived
      for model_version in model_versions:
        try:
          model_version = client.transition_model_version_stage(name=model_version.name,
                                                                version=model_version.version,
                                                                stage="Archived")
        except mlflow.exceptions.RestException:
          pass
    client.delete_registered_model(registry_model_name)
  else:
    print(f"No registered model named {registry_model_name} to delete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC * an **existing Delta table in Unity Catalog created/owned by current_user**
# MAGIC   * The data can be either batch scored data or inference logs a with the following **mandatory columns**
# MAGIC     * `timestamp` column _(TimeStamp)_ 
# MAGIC       * used for windowing/aggregation when calculating metrics
# MAGIC     * `model_version` column _(String)_
# MAGIC       * model version used for each prediction
# MAGIC     * `prediction` column _(String)_
# MAGIC       * model prediction output
# MAGIC     * _(OPTIONAL)_ `label` column _(String)_ 
# MAGIC       * ground truth label 
# MAGIC 
# MAGIC * _(OPTIONAL)_ Existing **baseline (Delta) table** to track model performance changes and feature distribution drifts
# MAGIC   * For model performance changes, we recommend using test/validation set
# MAGIC   * For feature distribution drifts, we recommend using training set or the associated feature tables 
# MAGIC   * These tables should contain (i) the same column names as the monitored tables and (ii) `model_version` column
# MAGIC   <br>
# MAGIC * * _(OPTIONAL)_ an existing _(dummy)_ model in MLflow's model registry (under `models:/registry_model_name`, for links to the monitoring UI and DBSQL dashboard)
# MAGIC   - Useful for visualizing Monitoring UI if the table is linked to an ML model in MLflow registry
# MAGIC   <br>
# MAGIC * _(RECOMMENDED)_ For all monitored tables, including baseline table(s) enable Delta's [Change-Data-Feed](https://docs.databricks.com/delta/delta-change-data-feed.html#enable-change-data-feed) table property for better metric computation performance

# COMMAND ----------

# MAGIC %md
# MAGIC To enable Change Data Feed, there are a few options:
# MAGIC <br> 
# MAGIC <br>
# MAGIC 1. At creation time
# MAGIC     - SQL: `TBLPROPERTIES (delta.enableChangeDataFeed = true)`
# MAGIC     - PySpark: `.option("delta.enableChangeDataFeed", "true")`
# MAGIC 1. Ad-hoc
# MAGIC     - SQL: `ALTER TABLE myDeltaTable SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`
# MAGIC 1. Set it in your notebook session:
# MAGIC     - `%sql set spark.databricks.delta.properties.defaults.enableChangeDataFeed = true;`

# COMMAND ----------

import mlflow
import sklearn

from datetime import timedelta, datetime
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F, types as T
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## User Journey
# MAGIC 1. Table Creation: Read raw input/features data and create training/inference sets
# MAGIC 2. Train a `baseline` model & register the model the MLflow Model Registry and transition to "Production"
# MAGIC 3. Generate predictions on test set and create **baseline table** with `model_version` defined as categorical/string
# MAGIC 4. Tune model and push new version to MLflow Registry and transition to "Staging"
# MAGIC 5. Generate predictions on `scoring_df1`. This would be the inference table.
# MAGIC 6. Define monitor on the inference table
# MAGIC 7. Simulate drifts in 3 relevant features, `scoring_df2`
# MAGIC 8. Generate predictions on the drifted dataset (`scoring_df2`) & update inference table
# MAGIC 9. Add/Join ground-truth labels to monitoring table and refresh monitor
# MAGIC 10. [Optional] Calculate custom metrics
# MAGIC 11. [Optional] Calculate thresholds
# MAGIC 
# MAGIC **Note:** if you already have an inference table and associated model(s) that fulfill the prerequisites listed above, you can skip steps 1-5 and 7-8

# COMMAND ----------

# Required parameters in order to run this notebook.
CATALOG = dbutils.widgets.get("monitor_catalog")
MODEL_NAME = dbutils.widgets.get("model_name") # Name of (registered) model
TABLE_NAME = dbutils.widgets.get("table_name")
QUICKSTART_MONITOR_DB = dbutils.widgets.get("monitor_db") # Output database/schema to store analysis/drift metrics tables in
BASELINE_TABLE = f"{MODEL_NAME}_baseline"  # OPTIONAL - Baseline table name, if any, for computing drift against baseline
TIMESTAMP_COL = "scoring_timestamp"
MODEL_VERSION_COL = "model_version"

# COMMAND ----------

# DBTITLE 1,Delete existing monitor (if exists)
try:
  # Using API call
  dm.delete_monitor(table_name=f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}", purge_artifacts=True)
except dm.errors.DataMonitoringError as e:
  if e.error_code == "MONITOR_NOT_FOUND" :
    print(f"No existing monitor on table {CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}!")
  elif e.error_code == "DATA_MONITORING_SERVICE_ERROR":
    print(e.message)
  else: 
    raise(e)

# COMMAND ----------

# DBTITLE 1,If user has permissions to create catalog (OPTIONAL)
# MAGIC %sql
# MAGIC -- CREATE CATALOG IF NOT EXISTS $monitor_catalog; -- If user has privileges to create one

# COMMAND ----------

# DBTITLE 1,Define Catalog (Mandatory) & set default database/schema to use
# MAGIC %sql
# MAGIC USE CATALOG $monitor_catalog;
# MAGIC CREATE SCHEMA IF NOT EXISTS $monitor_db;
# MAGIC USE $monitor_db;
# MAGIC DROP TABLE IF EXISTS $table_name; 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read dataset & prepare data
# MAGIC Dataset used for this example: [Airbnb price listing](http://insideairbnb.com/san-francisco/)

# COMMAND ----------

# Read data and add a unique id column (not mandatory but preferred)
raw_df = (spark.read.format("parquet")
          .load("/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/") 
          .withColumn("id", F.expr("uuid()"))
         )

display(raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Split data
# MAGIC Split data into a training set, baseline test table, and inference table. 
# MAGIC - The baseline test data will serve as the table with reference feature distributions.
# MAGIC - The inference table will then be split into two dataframes, `scoring_df1` and `scoring_df2`: they will function as new incoming batches for scoring. We will further simulate drifts on the `scoring_df`(s).

# COMMAND ----------

features_list = ["bedrooms", "neighbourhood_cleansed", "accommodates", "cancellation_policy", "beds", "host_is_superhost", "property_type", "minimum_nights", "bathrooms", "host_total_listings_count", "number_of_reviews", "review_scores_value", "review_scores_cleanliness"]

target_col = "price"

train_df, baseline_test_df, inference_df = raw_df.select(*features_list+[target_col]+["id"]).randomSplit(weights=[0.6, 0.2, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train a baseline random forest model

# COMMAND ----------

# Define the training datasets
X_train = train_df.drop("id", target_col).toPandas()
Y_train = train_df.select(target_col).toPandas().values.ravel()

# Define categorical preprocessor
categorical_cols = [col for col in X_train if X_train[col].dtype == "object"]
one_hot_pipeline = Pipeline(steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("onehot", one_hot_pipeline, categorical_cols)], remainder="passthrough", sparse_threshold=0)

# Define the model
skrf_regressor = RandomForestRegressor(
  bootstrap=True,
  criterion="absolute_error",
  max_depth=5,
  max_features=0.5,
  min_samples_leaf=0.1,
  min_samples_split=0.15,
  n_estimators=36,
  random_state=42,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", skrf_regressor),
])

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="random_forest_regressor") as mlflow_run:
    model.fit(X_train, Y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Register model and transition to Production

# COMMAND ----------

# clean up existing model
cleanup_registered_model(MODEL_NAME)

# Register model to MLflow Model Registry
run_id = mlflow_run.info.run_id
model_version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)

# COMMAND ----------

model_stage = "Production"

info = client.transition_model_version_stage(
  name=model_version.name,
  version=model_version.version,
  stage=model_stage,
  archive_existing_versions=True
)
print(info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create baseline table 
# MAGIC Please refer to the **FAQ section** in the [User Guide](https://drive.google.com/drive/u/0/folders/1oXuP-VleXmq0fTE4YovavboAVC7L-DF5) to understand how to best define the baseline table.

# COMMAND ----------

model_uri = f"models:/{MODEL_NAME}/{model_stage}"
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")

features = list(X_train.columns)
baseline_test_df_with_pred =(baseline_test_df
                             .withColumn("prediction", loaded_model(*features))
                             .withColumn(MODEL_VERSION_COL, F.lit(model_version.version)) )# Add model version column

display(baseline_test_df_with_pred)

# COMMAND ----------

# DBTITLE 1,Write table with CDF enabled
(baseline_test_df_with_pred
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema",True)
 .option("delta.enableChangeDataFeed", "true")
 .saveAsTable(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{BASELINE_TABLE}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Update model hyperparameter 
# MAGIC We will then push this model to Staging

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np

search_space = {
  "max_depth": hp.quniform("max_depth", 5, 30, 5),
  "n_estimators": hp.quniform("n_estimators", 30, 60, 5)
}

def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
  with mlflow.start_run(nested=True):
    # Split Train/Validation
    x_train, x_val,  y_train, y_val = train_test_split(X_train, Y_train, test_size=0.10)
                            
    # Define the model
    skrf_regressor = RandomForestRegressor(
      bootstrap=True,
      criterion="squared_error",
      max_depth=int(params["max_depth"]),
      max_features=0.5,
      min_samples_leaf=0.1,
      min_samples_split=0.15,
      n_estimators=int(params["n_estimators"]),
      random_state=42,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", skrf_regressor),
    ])
    
    # Fit
    model.fit(x_train, y_train)
                            
    # Evaluate (Mean-Squared-Error)
    validation_predictions = model.predict(x_val)
    mse_score = mean_squared_error(y_val, validation_predictions)
    mlflow.log_metric("mse", mse_score)
    
    # Set the loss to -mse_score so fmin minimizes the MSE
    return -mse_score

# Set parallelism
spark_trials = SparkTrials(parallelism=3) # A reasonable value for parallelism is the square root of max_evals.

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent.
with mlflow.start_run(run_name="rf_models"):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=9,
    trials=spark_trials,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Find best run and push new version to registry as Staging

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.mse DESC']).iloc[0]
new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", MODEL_NAME)

# COMMAND ----------

# Promote the new model version to Staging
client.transition_model_version_stage(
  name=MODEL_NAME,
  version=new_model_version.version,
  stage="Staging",
  archive_existing_versions=True
)

# COMMAND ----------

# DBTITLE 1,Load new model for batch scoring
new_loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{MODEL_NAME}/{new_model_version.version}", result_type="double")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Generate predictions on baseline test data with the new model _(OPTIONAL)_
# MAGIC In practice, you would do this if a new model version becomes the new baseline. You can also do this to track the model's metrics across different model versions.

# COMMAND ----------

(baseline_test_df
 .withColumn("prediction", new_loaded_model(*features))
 .withColumn(MODEL_VERSION_COL, F.lit(new_model_version.version))
 .write
 .format("delta")
 .mode("append")
 .saveAsTable(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{BASELINE_TABLE}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate predictions on incoming scoring data
# MAGIC 
# MAGIC ### Pre-Processing step for demo purposes
# MAGIC - Extract ground-truth labels
# MAGIC - Split into two batches

# COMMAND ----------

test_labels_df = inference_df.select("id", target_col)
scoring_df1, scoring_df2 = inference_df.drop(target_col).randomSplit(weights=[0.5, 0.5], seed=42)

# COMMAND ----------

# Simulate timestamp(s) if they don't exist
timestamp1 = (datetime.now() + timedelta(1)).timestamp()

pred_df1 = (scoring_df1
            .withColumn(TIMESTAMP_COL, F.lit(timestamp1).cast("timestamp")) 
            .withColumn("prediction", new_loaded_model(*features))
           )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Write scoring data with predictions out 
# MAGIC Add `model_version` column and write to monitored table

# COMMAND ----------

(pred_df1.withColumn(MODEL_VERSION_COL, F.lit(new_model_version.version)) 
       .write.format("delta").mode("overwrite") 
       .option("overwriteSchema",True) 
       .option("delta.enableChangeDataFeed", "true") 
       .saveAsTable(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Monitor
# MAGIC Using `InferenceLog` type analysis
# MAGIC 
# MAGIC **Required parameters**:
# MAGIC - `TABLE_NAME`: Name of the table to monitor.
# MAGIC - `PROBLEM_TYPE`: ML problem type, for `problem_type` parameter of `monitor_table`. 
# MAGIC   - Either `"classification"` or `"regression"`.
# MAGIC - `PREDICTION_COL`: Name of column in `TABLE_NAME` storing model predictions.
# MAGIC - `TIMESTAMP_COL`: Name of `timestamp` column in inference table
# MAGIC - `MODEL_VERSION_COL`: Name of column reflecting model version.
# MAGIC - `GRANULARITIES`: Monitoring analysis granularities 
# MAGIC   - E.g. `["5 minutes", "30 minutes", "1 hour", "1 day", "n weeks", "1 month", "1 year"]`
# MAGIC 
# MAGIC **Optional parameters**:
# MAGIC - `OUTPUT_SCHEMA_NAME`: _(OPTIONAL)_ Name of the database/schema where to create output tables (can be either {schema} or {catalog}.{schema} format). If not provided default/current DB will be used.
# MAGIC - `LINKED_ENTITIES` _(OPTIONAL but useful for Private Preview in order to visualize monitoring UI)_: List of Databricks entity names that are associated with this table. **Only following entities are supported:**
# MAGIC      - `["models:/registry_model_name", "models:/my_model"]` links model(s) in the MLflow registry to the monitored table.
# MAGIC 
# MAGIC **Monitoring parameters**:
# MAGIC - `BASELINE_TABLE_NAME` _(OPTIONAL)_: Name of table containing baseline data **NEEDS TO HAVE A `model_version` COLUMN** in case of `InferenceLog` analysis
# MAGIC - `SLICING_EXPRS` _(OPTIONAL)_: List of column expressions to independently slice/group data for analysis. (i.e. `slicing_exprs=["col_1", "col_2 > 10"]`)
# MAGIC - `CUSTOM_METRICS` _(OPTIONAL)_: A list of custom metrics to compute alongside existing aggregate, derived, and drift metrics.
# MAGIC - `SKIP_ANALYSIS` _(OPTIONAL)_: Flag to run analysis at monitor creation/update invoke time.
# MAGIC - `DATA_MONITORING_DIR` _(OPTIONAL)_: absolute path to existing directory for storing artifacts under `/{table_name}` (default=`/Users/{user_name}/databricks_data_monitoring`)
# MAGIC 
# MAGIC **Table parameters** :
# MAGIC - `LABEL_COL` _(OPTIONAL)_: Name of column storing labels
# MAGIC - `EXAMPLE_ID_COL` _(OPTIONAL)_: Name of (unique) identifier column (to be ignored in analysis)
# MAGIC 
# MAGIC **Make sure to drop any column that you don't want to track or which doesn't make sense from a business or use-case perspective**

# COMMAND ----------

help(dm.create_or_update_monitor)

# COMMAND ----------

PROBLEM_TYPE = dbutils.widgets.get("problem_type")  # ML problem type, one of "classification"/"regression"

# Validate that all required inputs have been provided
if None in [MODEL_NAME, PROBLEM_TYPE]:
    raise Exception("Please fill in the required information for model name and problem type.")

# Window sizes to analyze data over
GRANULARITIES = ["1 day"]                       

# Optional parameters to control monitoring analysis.
LABEL_COL = "price"  
SLICING_EXPRS = ["cancellation_policy", "accommodates > 2"]   # Expressions to slice data with
LINKED_ENTITIES = [f"models:/{MODEL_NAME}"]
# DATA_MONITORING_DIR = f"/Users/{username}/DataMonitoringTEST"

# Parameters to control processed tables.
PREDICTION_COL = "prediction"  # What to name predictions in the generated tables
EXAMPLE_ID_COL = "id" # Optional

# Custom Metrics
CUSTOM_METRICS = None

# COMMAND ----------

# DBTITLE 1,Create Monitor
print(f"Creating monitor for {TABLE_NAME}")

dm_info = dm.create_or_update_monitor(
    table_name=TABLE_NAME, # Or f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}"
    granularities=GRANULARITIES,
    analysis_type=analysis.InferenceLog(
        timestamp_col=TIMESTAMP_COL,
        example_id_col=EXAMPLE_ID_COL, # To drop from analysis
        model_version_col=MODEL_VERSION_COL, # Model version number 
        prediction_col=PREDICTION_COL,
        problem_type=PROBLEM_TYPE,
    ),
    baseline_table_name=BASELINE_TABLE,
    slicing_exprs=SLICING_EXPRS,
    linked_entities=LINKED_ENTITIES
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Inspect the analysis tables
# MAGIC 
# MAGIC Notice that the cell below shows that within the monitor_db, there are two other tables, in addition to the inference table:
# MAGIC 
# MAGIC 1. analysis_metrics
# MAGIC 2. drift_metrics
# MAGIC 
# MAGIC These two tables record the outputs of analysis jobs.

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES FROM $monitor_db LIKE '$table_name*'

# COMMAND ----------

# MAGIC %md
# MAGIC First, let's look at the `analysis_metrics` table.

# COMMAND ----------

analysis_df = spark.sql(f"SELECT * FROM {dm_info.assets.analysis_metrics_table_name}")
display(analysis_df)

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that for every column, the analysis table differentiates baseline data from scoring data and generates analyses based on:
# MAGIC - window
# MAGIC - model version
# MAGIC - slice key
# MAGIC 
# MAGIC We can also gain insight into basic summary statistics
# MAGIC - percent_distinct
# MAGIC - data_type
# MAGIC - min
# MAGIC - max
# MAGIC - etc.

# COMMAND ----------

display(analysis_df.filter("column_name='cancellation_policy'"))

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the drift table below, we are able to tell the shifts between the `train_df` and `scoring_df1`. 

# COMMAND ----------

display(spark.sql(f"SELECT column_name, * FROM {dm_info.assets.drift_metrics_table_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since this comparison of `scoring_df1` is made against the baseline `train_df`, we can see that `drift_type = "BASELINE"`. We will see another drift type, called `"CONSECUTIVE"` as we add more scored data.

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dm_info.assets.drift_metrics_table_name}").groupby("drift_type").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create data drifts(s) in 3 features
# MAGIC Simulate distributional changes for `neighbourhood_cleansed`, `cancellation_policy` and `accommodates`

# COMMAND ----------

display(scoring_df1.select(["neighbourhood_cleansed", "cancellation_policy", "accommodates"]))

# COMMAND ----------

remove_top_neighbourhood_list = ["South of Market", "Western Addition", "Downtown/Civic Center", "Bernal Heights", "Castro/Upper Market"]

scoring_df2_simulated = (scoring_df2
                      ### Remove top neighbourhoods to simulate change in distribution
                      .withColumn("neighbourhood_cleansed", 
                                  F.when(F.col("neighbourhood_cleansed").isin(remove_top_neighbourhood_list), "Mission")
                                  .otherwise(F.col("neighbourhood_cleansed")))
                      ### Introduce a new value to a categorical variable
                      .withColumn("cancellation_policy", 
                                  F.when(F.col("cancellation_policy")=="flexible", "super flexible")
                                  .otherwise(F.col("cancellation_policy")))
                      ### Replace all accommodates with 1
                      .withColumn("accommodates", F.lit(1).cast("double"))
                     )
display(scoring_df2_simulated.select(["neighbourhood_cleansed", "cancellation_policy", "accommodates"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Generate predictions on drifted observations & update inference tables
# MAGIC - Add the column `model_version`

# COMMAND ----------

# Simulate scoring that would happen in 2 days from now
timestamp2 = (datetime.now() + timedelta(2)).timestamp()
pred_df2 = (scoring_df2_simulated
            .withColumn("scoring_timestamp", F.lit(timestamp2).cast("timestamp")) 
            .withColumn("prediction", new_loaded_model(*features))
            .withColumn(MODEL_VERSION_COL, F.lit(new_model_version.version)) 
            .write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}")
           )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. (Ad-hoc) Join ground-truth labels to inference table

# COMMAND ----------

(spark.read.table(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}") 
     .join(test_labels_df, on="id", how="left") 
     .write.format("delta").mode("overwrite") 
     .option("mergeSchema",True) 
     .saveAsTable(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Update monitor to take label column into account

# COMMAND ----------

dm.update_monitor(table_name=TABLE_NAME,
                  updated_params={
                   "analysis_type":analysis.InferenceLog(
                      timestamp_col=TIMESTAMP_COL,
                      example_id_col=EXAMPLE_ID_COL,
                      model_version_col=MODEL_VERSION_COL,
                      prediction_col=PREDICTION_COL,
                      problem_type=PROBLEM_TYPE,
                      label_col=LABEL_COL 
                   )
                 })

# COMMAND ----------

# You can choose to refresh metrics now. But since we are going to add custom metrics below, we are going to refresh the monitor later.
# dm.refresh_metrics(table_name=TABLE_NAME,
#                    backfill=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Optional] 10. Refresh metrics by also adding custom metrics
# MAGIC Please refer to the **Custom Metrics** section in the [User Guide](https://drive.google.com/drive/u/0/folders/1oXuP-VleXmq0fTE4YovavboAVC7L-DF5)

# COMMAND ----------

from pyspark.sql import types as T
from databricks.data_monitoring.metrics import Metric
from math import exp

CUSTOM_METRICS = [
    Metric(
           metric_type="aggregate",
           metric_name="log_avg",
           input_columns=["price"],
           metric_definition="avg(log(abs(`{{column_name}}`)+1))",
           output_type=T.DoubleType()
           ),
    Metric(
           metric_type="derived",
           metric_name="exp_log",
           input_columns=["price"],
           metric_definition="exp(log_avg)",
           output_type=T.DoubleType()
        ),
    Metric(
           metric_type="drift",
           metric_name="delta_exp",
           input_columns=["price"],
           metric_definition="{{current_df}}.exp_log - {{base_df}}.exp_log",
           output_type=T.DoubleType()
        )
]

# COMMAND ----------

# DBTITLE 1,Update monitor
dm.update_monitor(table_name=TABLE_NAME,
                  updated_params={
                   "custom_metrics" : CUSTOM_METRICS
                 })

# COMMAND ----------

# MAGIC %md
# MAGIC #### Refresh metrics  & inspect dashboard
# MAGIC 
# MAGIC Inspect the auto-generated monitoring [DBSQL dashboard](https://docs.databricks.com/sql/user/dashboards/index.html).

# COMMAND ----------

dm.refresh_metrics(table_name=TABLE_NAME,
                   backfill=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inspect tables
# MAGIC 
# MAGIC Notice that after we updated the scoring data with a second batch, there are now several new rows in the analysis tables which correspond to the statistics per:
# MAGIC - input column
# MAGIC - input window
# MAGIC - model version
# MAGIC - different values of the slicing expressions
# MAGIC 
# MAGIC In particular, we can inspect the statistic for the total count of input rows in each scoring batch and in the baseline data. Notice that the respective counts correspond to `train_df`, `scoring_df1` and `scoring_df2`. In `scoring_df2`, the count has dropped due to our `string_indexer`'s `handleInvalid = "skip"` handling. 
# MAGIC ```
# MAGIC string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
# MAGIC ```

# COMMAND ----------

display(spark.sql(f"SELECT window, log_type, count, column_name, Model_Version, exp_log, log_avg, * FROM {dm_info.assets.analysis_metrics_table_name} WHERE COLUMN_NAME IN ('price') AND slice_key is NULL"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We can inspect the drift metrics in more detail. In the following query, `window_cmp` represents the window against which `window` is being compared. Also keep in mind that our setup simulates the following two scoring batches:
# MAGIC <br>
# MAGIC - scoring_df1 (-> pred_df) = tomorrow's date
# MAGIC - scoring_df2 (-> pred_df2) = the day after tomorrow
# MAGIC <br>
# MAGIC There is also the baseline data which corresponds to the training data of the model.
# MAGIC 
# MAGIC We can readily idenfity drifts using the table below! For instance, we can see the following drifts in column `accommodates` due to our simulation:
# MAGIC 1. pred_df vs. pred_df2  with KS statistic of 0.905 at p-value = 0
# MAGIC 2. pred_df2 vs. train_df_with_pred with KS statistic of 0.915 at p-value of 0
# MAGIC <br>
# MAGIC <br>
# MAGIC Unsurprisingly, the following comparison has no drift:
# MAGIC 1. train_df_with_pred vs scoring_df1 with KS statistic of 0.01 at p-value > 0.05

# COMMAND ----------

display(spark.sql(f"SELECT column_name, * FROM {dm_info.assets.drift_metrics_table_name} WHERE COLUMN_NAME IN ('accommodates', 'cancellation_policy', 'neighbourhood_cleansed') AND slice_key is NULL AND Model_Version = 2").drop("granularity", "Model_Version",  "slice_key", "slice_value", "non_null_columns_delta"))

# COMMAND ----------

display(spark.sql(f"SELECT window, window_cmp, column_name, model_version, delta_exp FROM {dm_info.assets.drift_metrics_table_name} WHERE COLUMN_NAME IN ('price') AND slice_key is NULL"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Optional] 11. Generate thresholds for alerting
# MAGIC 
# MAGIC You can set up alerts on the metrics in the analysis and drift tables using [Databricks SQL alerts](https://docs.databricks.com/sql/user/alerts/index.html). You need to first write a SQL query and then navigate to the SQL alerts page to create the alert.

# COMMAND ----------

thresh_pdf = dm.generate_thresholds(
        table_name=TABLE_NAME,
        model_version=new_model_version.version,
        column_metrics={
            "price": ["avg", "stddev", "avg_delta"],
            "cancellation_policy": ["avg_length", "chi_squared_test"]
        },
        confidence=0.95,
    )

# COMMAND ----------

display(thresh_pdf)
