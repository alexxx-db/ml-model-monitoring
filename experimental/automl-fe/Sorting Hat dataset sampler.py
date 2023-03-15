# Databricks notebook source
import pandas as pd

# COMMAND ----------

df_raw = pd.read_csv("/dbfs/FileStore/tables/data_train.csv")
df = df_raw[["Attribute_name", "y_act", "sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]]

# COMMAND ----------

df_raw

# COMMAND ----------

df_raw["y_act"].unique()

# COMMAND ----------

datetimes = df[df["y_act"] == "datetime"].sort_values("Attribute_name")
display(datetimes)

# COMMAND ----------

numerics = df[df["y_act"] == "numeric"].sort_values("Attribute_name")
display(numerics)

# COMMAND ----------

text = df[df["y_act"] == "sentence"].sort_values("Attribute_name")
display(text)

# COMMAND ----------

categorical = df[df["y_act"] == "categorical"].sort_values("Attribute_name")
display(categorical)

# COMMAND ----------


