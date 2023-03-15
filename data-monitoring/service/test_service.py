# Databricks notebook source
# MAGIC %md
# MAGIC #Tested CUJs
# MAGIC ## Read CUJs
# MAGIC 1) Create a monitor for a table owned by self
# MAGIC 2) Create a monitor for a table owned by others
# MAGIC 3) Create a 2nd monitor for the same table
# MAGIC 4) Create a monitor for non-existent UC table
# MAGIC ## Read CUJs
# MAGIC 1) Read a monitor created by self
# MAGIC 2) Read a monitor created by others
# MAGIC 3) Read a non-existent monitor
# MAGIC ## List CUJs
# MAGIC 1) List monitors created by self + others
# MAGIC ## Delete CUJs
# MAGIC 1) Delete a monitor created by self
# MAGIC 2) Delete a monitor created by others
# MAGIC 3) Delete a non-existent monitor

# COMMAND ----------

import requests
import json
import mlflow

# COMMAND ----------

TOKEN=mlflow.utils.databricks_utils._get_dbutils().entry_point.getDbutils().notebook().getContext().apiToken().get()
DOMAIN = "dbc-625b12b7-6534.dev.databricks.com"
table = "catalog.schema.table_shupengs"
table2 = "catalog.schema.table2_shupengs"
table3 = "catalog.schema.table3_shupengs"
others_table = "catalog.schema.alex_table"
non_existent_table = "catalog.schema.non_existent_table"
dashboard_id = "d5cf2acf-6a66-4826-8ff8-cb1cf0b28a63"

# COMMAND ----------

spark.range(10).write.format("delta").mode("overwrite").saveAsTable(table_name)
spark.range(10).write.format("delta").mode("overwrite").saveAsTable(table2_name)

# COMMAND ----------

# Helper method to generate the metadata in json
def gen_metadata_json(table):
    return {
        'table_name': table,
        'metadata': {
            'config': {
                'input_table_name': table,
                'granularities': ['1 day'],
                'linked_entities': ["models:/shupengs_test_model1"],
             },
             'status': 'MONITOR_STATUS_ACTIVE',
             'dashboard_id': dashboard_id,
             'analysis_metrics_table_name': 'catalog.schema.analysis_metrics_table_name_shupengs',
             'drift_metrics_table_name': 'catalog.schema.drift_metrics_table_name_shupengs',
             'client_version': '0.0.1',
         }
      }

# Helper method to delete a monitor
def delete_monitor(table):
    return requests.delete(
      f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{table}",
      headers={"Authorization": f"Bearer {TOKEN}"},
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read CUJs

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a monitor for a table owned by self

# COMMAND ----------

create_monitor_for_table = gen_metadata_json(table)

response = requests.post(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
  json=create_monitor_for_table
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a monitor for a table owned by others

# COMMAND ----------

create_monitor_for_others_table = gen_metadata_json(others_table)

response = requests.post(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{others_table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
  json=create_monitor_for_others_table
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a 2nd monitor for the same table

# COMMAND ----------

response = requests.post(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
  json=create_monitor_for_table
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a monitor for non-existent UC table

# COMMAND ----------

create_payload_for_non_existent_table = gen_metadata_json(non_existent_table)


response = requests.post(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{non_existent_table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
  json=create_payload_for_non_existent_table
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Read CUJs

# COMMAND ----------

# MAGIC %md
# MAGIC ###Read a monitor created by self

# COMMAND ----------

response = requests.get(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Read a monitor created by others

# COMMAND ----------

response = requests.get(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{others_table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Read a non-existent monitor

# COMMAND ----------

response = requests.get(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{table2}",
  headers={"Authorization": f"Bearer {TOKEN}"},
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ## List CUJs

# COMMAND ----------

# MAGIC %md
# MAGIC ###List monitors created by self + others

# COMMAND ----------

# MAGIC %scala
# MAGIC import java.net.URLEncoder
# MAGIC val entityName = "models:/shupengs_test_model1"
# MAGIC val encodedName = URLEncoder.encode(entityName, "UTF-8")

# COMMAND ----------

response = requests.get(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/entities/models%3A%2Fshupengs_test_model1",
  headers={"Authorization": f"Bearer {TOKEN}"},
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Delete CUJs

# COMMAND ----------

# MAGIC %md
# MAGIC ###Delete a monitor created by self

# COMMAND ----------

response = requests.delete(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
)
response.json()

# COMMAND ----------

response = requests.get(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Delete a monitor created by others

# COMMAND ----------

response = requests.delete(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{others_table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
)
response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Delete a non-existent monitor

# COMMAND ----------

response = requests.delete(
  f"https://{DOMAIN}/api/2.0/data-monitoring-md/tables/{non_existent_table}",
  headers={"Authorization": f"Bearer {TOKEN}"},
)
response.json()
