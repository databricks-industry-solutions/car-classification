# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import yaml
with open('config/application.yaml', 'r') as f:
  config = yaml.safe_load(f)

# COMMAND ----------

dbutils.fs.mkdirs(config['database']['path'])
_ = sql("CREATE DATABASE IF NOT EXISTS {} LOCATION '{}'".format(
  config['database']['name'], 
  config['database']['path']
))

_ = sql("USE {}".format(config['database']['name']))

# COMMAND ----------

# Set mlflow experiment explicitly to make sure the code runs in both interactive execution and job execution
import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/car_classification'.format(username))
