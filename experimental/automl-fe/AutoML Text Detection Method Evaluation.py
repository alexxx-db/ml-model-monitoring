# Databricks notebook source
# MAGIC %pip install langdetect fasttext scikit-learn==1.0

# COMMAND ----------

import fasttext
from langdetect import detect, LangDetectException

LANGDETECT_SAMPLE_SIZE = 1000
FASTTEXT_MODEL_PATH = "/dbfs/FileStore/alkis/lid_176.ftz"
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# calculate percent distinct
def calc_percent_distinct(vals):
  return (vals.nunique() * 1.0 / vals.size)

# return confidence of fasttext model that string is english
def is_english_fasttext(s):
  langs, confs = ft_model.predict(s.replace("\n", " "))
  
  try:
    english_idx = langs.index("__label__en")
    return confs[english_idx]
  except ValueError:
    return 0.0

# use fasttext model on every non-null row and average confidences
def detect_fasttext_by_rows(col):
  vals = col.dropna()
  percent_distinct = calc_percent_distinct(vals)
  
  confs = vals.apply(lambda s: is_english_fasttext(s))
  return percent_distinct, confs.mean()

# concatenate all rows into a single string and detect
def detect_fasttext_by_concat(col):
  vals = col.dropna()
  percent_distinct = calc_percent_distinct(vals)
  
  concatenated = vals.astype(str).str.cat(sep=" ")
  return percent_distinct, is_english_fasttext(concatenated)


# use langdetect built-in detect() method
def is_english_langdetect(text):
  try:
    if detect(text) != "en":
      return False
  except LangDetectException:
    return False
  return True

# use langdetect on every non-null row and return the fraction detected to be english
def detect_langdetect_by_sample(col):
  vals = col.dropna()
  percent_distinct = calc_percent_distinct(vals)
    
  if vals.size > LANGDETECT_SAMPLE_SIZE:
    vals = vals.sample(LANGDETECT_SAMPLE_SIZE)

  num_english = vals.apply(lambda s: is_english_langdetect(s)).sum()
  conf = (num_english * 1.0) / vals.size if num_english > 0 else 0.0
  return percent_distinct, conf

# COMMAND ----------

import pandas as pd

SORTINGHAT_CSV_FILES = "/dbfs/viswesh.periyasamy/type-inference-benchmark/RawCSVFiles/"
TWITTER_FILE = "/dbfs/viswesh.periyasamy/type-inference-benchmark/twitter/all_annotated.tsv"

# English datasets picked from sortinghat benchmarks
# (csv_file_path, text_column)
english_datasets = [
  ("fromWeb.csv", "Lyrics"),
  ("new.csv", "Message"),
  ("Restaurant_Reviews.csv", "Review"),
  ("glassdoorreviews.csv", "Reviews"),                                           # <-- large proportion of repeat values
  ("housing-community-investment-service-locations.csv", "SERVICE DESCRIPTION"), # <-- large proportion of repeat values
  #("Subnational-PovertySeries.csv", "Long definition"), <-- removed because we only have 3 rows
  ("transcripts.csv", "transcript"),
  ("filmes_denzel_V2.csv", "Sinopse"),
  ("SB272_Systems List.csv", "Statement Of Purpose"),
  #("dishes and recipes.csv", "Steps"),                  <-- removed because we only have 9 rows
  ("hotel_reviews.csv", "reviews_text"),
  ("u_Womens Clothing E-Commerce Reviews.csv", "Review Text"),
  ("ahca_polls.csv", "Text"),
  ("debate.csv", "Text"),
  ("e_monster_com-job_sample.csv", "job_description"),
  ("oreo_rankings.csv", "notes_and_discussion"),
  ("us_mass_shootings.csv", "summary"),
]

# Categorical (non-text) datasets picked from sortinghat benchmarks
non_text_datasets = [
  ("SF_Park_Scores.csv", "Facility Type"),
  ("IHMStefanini_industrial_safety_and_health_database.csv", "Genre"),  # binary, "Male" or "Female"
  ("Family Income and Expenditure.csv", "Household Head Occupation"),
  ("FAO.csv", "Item"),
  ("U.S._Chronic_Disease_Indicators.csv", "LocationDesc"),
  ("crime_incident_data2014.csv", "Major Offense Type"),
  ("2016.csv", "Region"),
  ("PPD_PlanLevel_2017_12_11.csv", "SocSecCovered_verbatim"),
  ("pax_20_02_2018_1_CSV.csv", "Status"),
  ("Washington_State_HDMA-2016.csv", "action_taken_name"),
]

# filter twitter dataset into non-english rows, then choose 5 countries with most number of rows
# BR    945
# ID    707
# JP    367
# TR    304
# AR    291
non_english_datasets = ["BR", "ID", "JP", "TR", "AR"]


# calculate size of each dataset
datasets = []
sizes = []
for dataset, text_col_name in english_datasets + non_text_datasets:
  datasets.append(dataset)
  df = pd.read_csv(SORTINGHAT_CSV_FILES + dataset, usecols=[text_col_name])
  sizes.append(len(df))
  
twitter_df = pd.read_csv(TWITTER_FILE, sep="\t")
non_english_df = twitter_df[twitter_df["Definitely Not English"] == 1]
for country in non_english_datasets:
  datasets.append(f"Twitter {country}")
  df = non_english_df[non_english_df["Country"] == country]
  sizes.append(len(df))

df = pd.DataFrame({"datasets": datasets, "size": sizes}).sort_values("size")
display(df)
ax = df.plot.bar(x="datasets", y="size", figsize=(12, 6))

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.metrics import precision_recall_fscore_support, ConfusionMatrixDisplay

DETECTION_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
PERCENT_DISTINCT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

EVAL_FNS = [
  (detect_fasttext_by_rows, "fasttext row average"),
  (detect_fasttext_by_concat, "fasttext row concat"),
  (detect_langdetect_by_sample, "langdetect sample"),
]
  
results = {}
for eval_fn, eval_name in EVAL_FNS:
  result = {}
  total_duration = 0.0
  y_true = []
  confs = []
  percent_distincts = []
  # run on known english columns and non-text columns
  for dataset, text_col_name in english_datasets + non_text_datasets:
    df = pd.read_csv(SORTINGHAT_CSV_FILES + dataset, usecols=[text_col_name])
    text_col = df[text_col_name]

    start = perf_counter()
    percent_distinct, conf = eval_fn(text_col)
    duration = perf_counter() - start

    confs.append(conf)
    percent_distincts.append(percent_distinct)
    y_true.append((dataset, text_col_name) in english_datasets) # True if this is an english dataset, else False
    total_duration += duration

  # run on known non-english columns
  twitter_df = pd.read_csv(TWITTER_FILE, sep="\t", usecols=["Definitely Not English", "Country", "Tweet"])
  non_english_df = twitter_df[twitter_df["Definitely Not English"] == 1]
  for country in non_english_datasets:
    df = non_english_df[non_english_df["Country"] == country]
    text_col = df["Tweet"]

    start = perf_counter()
    percent_distinct, conf = eval_fn(text_col)
    duration = perf_counter() - start

    confs.append(conf)
    percent_distincts.append(percent_distinct)
    y_true.append(False)
    total_duration += duration

  result["total_duration"] = total_duration
  result["y_true"] = y_true
  result["confs"] = confs
  result["percent_distincts"] = percent_distincts
  
  results[eval_name] = result


for detection_threshold in DETECTION_THRESHOLDS:
  for percent_distinct_threshold in PERCENT_DISTINCT_THRESHOLDS:
    print(f"////////////////// PERCENT_DISTINCT_THRESHOLD = {percent_distinct_threshold},  DETECTION_THRESHOLD = {detection_threshold} //////////////////")
    for eval_name, result in results.items():
      print(f"Detection using {eval_name}")
      y_true = result["y_true"]
      confs = result["confs"]
      percent_distincts = result["percent_distincts"]
      total_duration = result["total_duration"]

      # classify each point based on our current thresholds
      y_pred = []
      for percent_distinct, conf in zip(percent_distincts, confs):
        if percent_distinct < percent_distinct_threshold or conf < detection_threshold:
          y_pred.append(False)
        else:
          y_pred.append(True)

      # calculate metrics
      precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, beta=1, pos_label=True, average="binary")
      print(f"precision: {precision}")
      print(f"recall: {recall}")
      print(f"f1 score: {f1_score}")
      print(f"total duration: {total_duration} seconds")

      # plot confusion matrix
      fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
      ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axs[0])

      # plot distinct percentages and confidences
      axs[1].scatter(confs, percent_distincts, c=y_true, cmap="hot_r", vmin=-2)
      axs[1].axvline(detection_threshold, color="b")
      axs[1].axhline(percent_distinct_threshold, color="g")
      axs[1].set_xlabel("confidences")
      axs[1].set_ylabel("percent_distinct")
      plt.show()
      print("-------------------------------------------")  

# COMMAND ----------


