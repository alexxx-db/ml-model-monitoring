# Databricks notebook source
import pandas as pd

# COMMAND ----------

df_raw = pd.read_csv("/dbfs/FileStore/tables/data_train.csv")
df = df_raw[["Attribute_name", "y_act", "total_vals", "num_of_dist_val", "sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]]

# COMMAND ----------

df_raw

# COMMAND ----------

def check_numeric(x):
  try:
    float(x)
    return True
  except ValueError:
    return False

# COMMAND ----------

# columns with true label 'categorical' that can be converted to numeric
categoricals = df[(
  df["sample_1"].apply(check_numeric)
) & (
  df["sample_2"].apply(check_numeric)
) & (
  df["y_act"] == "categorical"
)]
categoricals

# COMMAND ----------

# columns with true label 'numeric' that can be converted to numeric
numerics = df[(
  df["sample_1"].apply(check_numeric)
) & (
  df["sample_2"].apply(check_numeric)
) & (
  df["y_act"] == "numeric"
)]
numerics

# COMMAND ----------

import operator
from statistics import mean

import pandas as pd

def detect_type(row, cat_card_thresh):
  sample_cols = ["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]
  try:
    for sample in sample_cols:
      row[sample] = pd.to_numeric(row[sample], errors="raise")

    cardinality = row["num_of_dist_val"]
    if cardinality < cat_card_thresh:
      return "categorical"
    else:
      return "numeric"

  except ValueError:
    return "categorical"

# TUNE CATEGORICAL CARDINALITY THRESHOLD
accuracies = {}
for cat_card_thresh in range(10, 210, 10):    
  print("---------------------------------")
  print(f"CATEGORICAL_CARDINALITY_THRESHOLD: {cat_card_thresh}")

  correct = 0
  for _, row in categoricals.iterrows():
    prediction = detect_type(row, cat_card_thresh)
    if prediction == "categorical":
      correct += 1

  cat_accuracy = correct * 1.0 / len(categoricals)
  print(f"categorical correct: {correct}")
  print(f"categorical accuracy: {cat_accuracy}")

  correct = 0
  for _, row in numerics.iterrows():
    prediction = detect_type(row, cat_card_thresh)
    if prediction == "numeric":
      correct += 1

  num_accuracy = correct * 1.0 / len(numerics)
  print(f"numeric correct: {correct}")
  print(f"numeric accuracy: {num_accuracy}")
  accuracies[cat_card_thresh] = (cat_accuracy, num_accuracy)
    
    
best_avg_threshold = max(accuracies, key=lambda t: mean(accuracies[t]))
best_min_threshold = max(accuracies, key=lambda t: min(accuracies[t]))
print()
print(f"CATEGORICAL_CARDINALITY_THRESHOLD with highest avg accuracy: {best_avg_threshold}\naccuracies: {accuracies[best_avg_threshold]}")
print(f"CATEGORICAL_CARDINALITY_THRESHOLD with highest min accuracy: {best_min_threshold}\naccuracies: {accuracies[best_min_threshold]}")

# COMMAND ----------

# MAGIC %md Choose `CATEGORICAL_CARDINALITY_THRESHOLD = 40` to maintain best minimum accuracy on both numeric and categorical data

# COMMAND ----------

# TUNE STRONG NUMERIC DETECTION THRESHOLD

CATEGORICAL_CARDINALITY_THRESHOLD = 40
def strongly_detect_numeric(row, num_card_thresh):
  sample_cols = ["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]
  try:
    for sample in sample_cols:
      row[sample] = pd.to_numeric(row[sample], errors="raise")

    cardinality = row["num_of_dist_val"]
    if cardinality < CATEGORICAL_CARDINALITY_THRESHOLD:
      return "categorical"
    
    # now in weak detection territory
    samples = row[sample_cols]
    if is_float_dtype(sample) and np.modf(samples.to_numpy())[0].any():
      return "numeric"
    
    if cardinality >= num_card_thresh:
      return "numeric"
    else:
      "categorical"
  except ValueError:
    return "categorical"
  
false_positive_rates = {}
for num_card_thresh in range(CATEGORICAL_CARDINALITY_THRESHOLD, 1000, 10):    

  print("---------------------------------")
  print(f"STRONG_NUMERIC_DETECTION_THRESHOLD: {num_card_thresh}")

  wrong = 0
  for _, row in categoricals.iterrows():
    prediction = detect_type(row, num_card_thresh)
    if prediction == "numeric":
      wrong += 1

  false_positive_rate = wrong * 1.0 / len(categoricals)
  print(f"categorical wrong: {wrong}")
  print(f"categorical false positive rate: {false_positive_rate}")

  false_positive_rates[num_card_thresh] = false_positive_rate
    
fpr_sorted = list(false_positive_rates.items())
fpr_sorted.sort(key = lambda x: x[1], reverse=True)
print()
print("STRONG_NUMERIC_DETECTION_THRESHOLD with best false positive rate on strictly categorical data")
for fpr in fpr_sorted:
  print(fpr)

# COMMAND ----------

# MAGIC %md Choose `STRONG_NUMERIC_DETECTION_THRESHOLD = 250` to maintain categorical false positive rate of ~5%

# COMMAND ----------

# TUNE STRONG CATEGORICAL DETECTION THRESHOLD

CATEGORICAL_CARDINALITY_THRESHOLD = 40
def strongly_detect_categorical(row, strong_cat_card_thresh):
  sample_cols = ["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]
  try:
    for sample in sample_cols:
      row[sample] = pd.to_numeric(row[sample], errors="raise")

    cardinality = row["num_of_dist_val"]
    if cardinality >= CATEGORICAL_CARDINALITY_THRESHOLD:
      return "numeric"
    
    # now in weak detection territory
    if cardinality <= strong_cat_card_thresh:
      return "categorical"
    else:
      "numeric"
  except ValueError:
    return "categorical"
  
false_positive_rates = {}
for strong_cat_card_thresh in range(0, CATEGORICAL_CARDINALITY_THRESHOLD):    

  print("---------------------------------")
  print(f"STRONG_CATEGORICAL_DETECTION_THRESHOLD: {strong_cat_card_thresh}")

  wrong = 0
  for _, row in numerics.iterrows():
    prediction = detect_type(row, strong_cat_card_thresh)
    if prediction == "categorical":
      wrong += 1

  false_positive_rate = wrong * 1.0 / len(numerics)
  print(f"numeric wrong: {wrong}")
  print(f"numeric false positive rate: {false_positive_rate}")

  false_positive_rates[strong_cat_card_thresh] = false_positive_rate
    
fpr_sorted = list(false_positive_rates.items())
fpr_sorted.sort(key = lambda x: x[1], reverse=True)
print()
print("STRONG_CATEGORICAL_DETECTION_THRESHOLD with best false positive rate on strictly numeric data")
for fpr in fpr_sorted:
  print(fpr)

# COMMAND ----------

# MAGIC %md Choose `STRONG_CATEGORICAL_DETECTION_THRESHOLD = 10` to maintain numeric false positive rate of ~5%

# COMMAND ----------

import numpy as np

from sklearn.metrics import roc_curve, auc


# STRONG_CATEGORICAL_DETECTION_THRESHOLD = 10

# def detect_strong_categorical(row):
#   sample_cols = ["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]
#   try:
#     for sample in sample_cols:
#       row[sample] = pd.to_numeric(row[sample], errors="raise")

#     cardinality = row["num_of_dist_val"]
#     if cardinality <= STRONG_CATEGORICAL_DETECTION_THRESHOLD:
#       return "categorical"
#     else:
#       return "numeric"
#   except ValueError:
#     return "categorical"

predictions = []
labels = []
for _, row in categoricals.iterrows():
  #predictions.append(detect_strong_categorical(row))
  predictions.append(1.0 - (1.0/row["num_of_dist_val"]))
  labels.append("categorical")
for _, row in numerics.iterrows():
  #predictions.append(detect_strong_categorical(row))
  predictions.append(1.0 - (1.0/row["num_of_dist_val"]))
  labels.append("numeric")

#predictions = [1 if p == "categorical" else 0 for p in predictions]
labels = [0 if l == "categorical" else 1 for l in labels]

fpr, tpr, thresh = roc_curve(labels, predictions)
auroc = auc(fpr, tpr)

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(12,9), dpi=80)
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auroc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(np.arange(0, 1.0, 0.05))
plt.yticks(np.arange(0, 1.0, 0.05))

plt.axvline(x=0.05)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for strong detection thresholds')
plt.legend(loc="lower right")
plt.show()

# COMMAND ----------


