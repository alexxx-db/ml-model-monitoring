-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Model Monitoring Usage Data Pipeline

-- COMMAND ----------

USE DATABASE ml_data_db

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Extract MM non-DLT revenue from `prod.workload_insights`

-- COMMAND ----------

-- This assumes DLT workloads are not added into the workload_insights table yet. Once they are added (https://databricks.atlassian.net/browse/ES-331156), we do not have to use prod.workloads_cluster_agg to compute DLT revenue, and can solely rely on workload_insights table to compute all Model Monitoring related revenue.
CREATE OR REPLACE VIEW non_dlt_daily_stats AS
SELECT 
  date,
  cloudType,
  customerType,
  canonicalCustomerName,
  workspaceId,
  COUNT(1) AS num_api_calls,
  SUM(IF(workloadName = "commandRuns", attributedRevenue * 1.2, attributedRevenue)) AS revenue
FROM prod.workload_insights
-- '2022-03-22' is the Model Monitoring private preview launch date.
WHERE date >= "2022-03-22"
AND ((workloadName = "commandRuns" AND workloadTags.commandLanguage = "python") OR (workloadName = "jobRuns" AND workloadTags.numPythonCommands > 0))
AND array_contains(packages.pythonPackages, "databricks.model_monitoring")
GROUP BY 1, 2, 3, 4, 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Extract MM DLT revenue from `prod.workloads_cluster_agg`

-- COMMAND ----------

CREATE OR REPLACE VIEW dlt_daily_revenue AS
SELECT 
  wca.date,  
  wca.cloudType,
  wca.customerType,
  wca.canonicalCustomerName,
  wca.workspaceId,
  -- Fallback logic to decide the revenue share price. For Microsoft, we have a discount ratio (0.85) when using the list price.
  SUM(wca.dbus * IF(er.revSharePrice IS NOT NULL, er.revSharePrice, IF(wca.cloudType != "azure", wca.listPrice, wca.listPrice * 0.85))) AS revenue
FROM
(
  SELECT
    date, tags.clusterId
  FROM prod.usage_logs
  WHERE metric = "modelMonitoringEvent"
  GROUP BY 1, 2
) AS ul
JOIN prod.workloads_cluster_agg AS wca
ON (wca.date = ul.date AND wca.clusterId = ul.clusterId)
LEFT JOIN prod.effective_rates er ON (wca.date = er.date AND wca.workspaceId = er.workspaceId AND wca.sku = er.sku)
WHERE wca.sku LIKE '%DLT%'
GROUP BY 1, 2, 3, 4, 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Sum up DLT and non-DLT revenue and materialize the data into mm_revenue table

-- COMMAND ----------

CREATE OR REPLACE TABLE mm_daily_stats AS
SELECT 
  date,
  date_sub(trunc(date_add(date, 2), 'WEEK'), 2) AS week_start,
  cloudType,
  customerType,
  canonicalCustomerName,
  workspaceId,
  num_api_calls,
  non_dlt_revenue,
  dlt_revenue,
  revenue
FROM (  
  SELECT 
    coalesce(non_dlt.date, dlt.date) AS date,
    coalesce(non_dlt.cloudType, dlt.cloudType) AS cloudType,
    coalesce(non_dlt.customerType, dlt.customerType) AS customerType,
    coalesce(non_dlt.canonicalCustomerName, dlt.canonicalCustomerName) AS canonicalCustomerName,
    coalesce(non_dlt.workspaceId, dlt.workspaceId) AS workspaceId,
    coalesce(non_dlt.revenue, 0) AS non_dlt_revenue,
    coalesce(non_dlt.num_api_calls, 0) AS num_api_calls,
    coalesce(dlt.revenue, 0) AS dlt_revenue,
    coalesce(non_dlt.revenue, 0) + coalesce(dlt.revenue, 0) AS revenue
  FROM non_dlt_daily_stats non_dlt
  FULL JOIN dlt_daily_revenue dlt ON (non_dlt.date = dlt.date AND non_dlt.workspaceId = dlt.workspaceId)
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Extract MM FE usage from `prod.usage_logs`

-- COMMAND ----------

CREATE OR REPLACE TABLE mm_fe_usage_logs AS
SELECT 
  ul.tags.pseudoUserId as user, 
  ul.date, 
  date_sub(trunc(date_add(ul.date, 2), 'WEEK'), 2) AS week_start,
  ul.tags.eventName as eventName, 
  ul.workspaceId,
  pw.canonicalCustomerName,
  pw.cloudType
FROM prod.usage_logs ul
JOIN prod.workspaces pw ON (ul.workspaceId = pw.workspaceId)
WHERE ul.date > '2022-03-22'
AND ul.metric = "clientsideEvent"
AND ul.tags.eventType = "modelMonitoringEvent"

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Extract MM Python Usage from `prod.usage_logs`

-- COMMAND ----------

CREATE OR REPLACE VIEW mm_python_usage_logs_raw_events
AS (
SELECT *,
  COALESCE(tags.commandRunId, concat(workspaceId, "-", tags.jobId, "-", COALESCE(tags.idInJob, "1"))) as workloadId
FROM   prod.usage_logs
WHERE  metric = "modelMonitoringEvent"
       AND date >= date("2022-03-22")
);

CREATE OR REPLACE TABLE ml_data_db.mm_python_usage_logs
USING DELTA PARTITIONED BY (date) AS (
  SELECT raw.*, 
    workspaceName, 
    customerType, 
    cloudType,
    isRealCustomer, 
    canonicalCustomerId,
    canonicalCustomerName
  FROM mm_python_usage_logs_raw_events raw
    JOIN  prod.workspaces
    USING (workspaceId) 
);
