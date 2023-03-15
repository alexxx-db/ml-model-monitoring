# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Model Monitoring Bug Bash UJ2
# MAGIC 
# MAGIC Please make a copy of this notebook before use.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC Useful links:
# MAGIC  - [Bug Bash Doc](http://go/modmon/bugbash)
# MAGIC  - [Quickstart Notebook](https://e2-dogfood.staging.cloud.databricks.com/?o=6051921418418893#notebook/2390739687257529/command/2133117006775689)
# MAGIC  - [API Reference](https://docs.google.com/document/d/1L_fqZzz9ABx2E10NMW8Pg0EdtKpxJxAwJNJiVCaCgCs/edit#heading=h.x3x9cjfy4srj)
# MAGIC  - [User Guide](https://docs.google.com/document/d/1pWUEPY7vF80BSrQp_cqGjowo7Zn91_VvuoZ6J-ZeUKk/edit#heading=h.pc8x54hv4i93)

# COMMAND ----------

# Install model monitoring client library. Eventually this will be included in MLR
%pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/model-monitoring/e7394149-4c48-42a3-94af-11cee8964415/databricks_model_monitoring-0.0.0.dev0-py3-none-any.whl"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Create the monitor
# MAGIC 
# MAGIC We have registered a classification model [mm_bug_bash](https://e2-dogfood.staging.cloud.databricks.com/?o=6051921418418893#mlflow/models/mm_bug_bash) for you. 
# MAGIC 
# MAGIC The model is trained with Census Income csv dataset `/dbfs/databricks-datasets/adult/adult.data` and the ground-truth column is `income`. If you want to explore the dataset, you can use following code snippet to load it:
# MAGIC ```
# MAGIC import pandas as pd
# MAGIC cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
# MAGIC         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
# MAGIC         'hours-per-week', 'native-country', 'income']
# MAGIC pdf = pd.read_csv(
# MAGIC   "/dbfs/databricks-datasets/adult/adult.data",
# MAGIC   names=cols,
# MAGIC   skipinitialspace=True,
# MAGIC )
# MAGIC df = spark.createDataFrame(pdf)
# MAGIC ```

# COMMAND ----------

# We recommend to use your name in the monitor name to avoid the conflict with other people.
monitor_name = "{your_name}_monitor"
model_name = "mm_bug_bash"

assert(monitor_name != "{your_name}_monitor")

# COMMAND ----------

from pyspark.sql import types as T
import databricks.model_monitoring as mm

mm_info = mm.create_monitor(
  model_name=model_name,
  monitor_name=monitor_name,
  model_type="classifier",
  # This specifies that metrics will be computed based on daily windows
  granularities=["1 day"],

  # The column used to store model predictions
  prediction_col="income_predicted",
  # The column used to store ground truth labels
  label_col="income",
  # The column used to group data into time windows
  timestamp_col="dt",
  # The example id column uniquely identifies a row to link ground truth labels that arrive later
  id_cols=["example_id"],
  # Additional fields to track in the Logging Table.
  extra_logging_fields=[T.StructField("example_id", T.StringType(), False)],
  
  # List of column expressions to slice data with for targeted analysis.
  slicing_exprs=["age > 18", "race"],
  # Name of the existing database to store logged scoring data and analyses.
  database_name="mm_bug_bash_db"
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Log the data
# MAGIC 
# MAGIC The following two CSV fils are the training and testing dataset for the [mm_bug_bash](https://e2-dogfood.staging.cloud.databricks.com/?o=6051921418418893#mlflow/models/mm_bug_bash). You want to use them for the current UJ.
# MAGIC  - `/dbfs/databricks-datasets/adult/adult.data`
# MAGIC  - `/dbfs/databricks-datasets/adult/adult.test`
# MAGIC  
# MAGIC  
# MAGIC ### Please Note:
# MAGIC  - The ground-truth column of the testing dataset format is not consistent with the training data. The following code snippet can help you.
# MAGIC  ```
# MAGIC  import pandas as pd
# MAGIC  cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
# MAGIC            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
# MAGIC            'hours-per-week', 'native-country', 'income']
# MAGIC  test_pdf = pd.read_csv(
# MAGIC   "/dbfs/databricks-datasets/adult/adult.test",
# MAGIC   names=cols,
# MAGIC   skipinitialspace=True,
# MAGIC )
# MAGIC # We need to clean this dataset to make its labels consistent with training data
# MAGIC test_pdf["income"].replace(
# MAGIC   {'<=50K.' : '<=50K',
# MAGIC    '>50K.'  : '>50K'},
# MAGIC   inplace=True
# MAGIC )
# MAGIC test_df = spark.createDataFrame(test_pdf)
# MAGIC ```
# MAGIC  - You may see the logging failed if you try to log the data directly. Don't worry, try to massage the data based on the `mm.create_monitor` call above and the error messages.
# MAGIC     - Hint: you can use the sql function `uuid()` to generate a unique string per row, e.g.,:
# MAGIC     ```
# MAGIC     import pyspark.sql.functions as F
# MAGIC     df=df.withColumn("a", F.expr("uuid()"))
# MAGIC     ```

# COMMAND ----------

# Log Baseline Data
mm.log_baseline_data(
  # UJ: Fill the right parameters to log the data.
)

# COMMAND ----------

# Log Scored Data
mm.log_scored_data(
  # UJ: Fill the right parameters to log the data.
)

mm.log_labels(
   # UJ: Fill the right parameters to log the labels.
)

# COMMAND ----------

# UJ: Please check that data is logged as expected by querying the logging table.
# Hint: You can find the logging table by calling `get_monitor_info()`.


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Trigger Analysis and Inspect Results
# MAGIC 
# MAGIC Please trigger the analysis pipeline to process the data you just logged and then check that the pipeline generates output in the analysis tables.
# MAGIC 
# MAGIC Hint: You can can discover the analysis pipeline and the output tables in the "Monitoring" tab of the registered model, or by calling `get_monitor_info()`.

# COMMAND ----------

# UJ: Please inspect the analysis pipeline output tables.

