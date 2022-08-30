# Databricks notebook source
# import libraries
import pyspark.sql.functions as F
from pyspark.sql.types import *
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector, SparseVector

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression


# COMMAND ----------

# Azure blob storage init script
blob_container = "airline-data" # The name of your container created in https://portal.azure.com
storage_account = "w261team01" # The name of your Storage account created in https://portal.azure.com
secret_scope = "sas_team01" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "sas_key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# inspect folders in blob storage
display(dbutils.fs.ls(f"{blob_url}"))

# COMMAND ----------

# columns for filtering omni data to just columns needed for modeling
# include = ['utc2H',
#  'OP_CARRIER_FL_NUM',
#  'OP_UNIQUE_CARRIER',
#  'YEAR',
#  'QUARTER',
#  'MONTH',
#  'DAY_OF_MONTH',
#  'DAY_OF_WEEK',
#  'DISTANCE_GROUP',
#  'dep_hour',
#  'DEST_TS_2H',
#  'DEST_ICE_2H',
#  'DEST_SNOW_2H',
#  'DEST_FOG_2H',
#  'DEST_TS_4H',
#  'DEST_ICE_4H',
#  'DEST_SNOW_4H',
#  'DEST_FOG_4H',
#  'ORIGIN_TS_2H',
#  'ORIGIN_ICE_2H',
#  'ORIGIN_SNOW_2H',
#  'ORIGIN_FOG_2H',
#  'ORIGIN_TS_4H',
#  'ORIGIN_ICE_4H',
#  'ORIGIN_SNOW_4H',
#  'ORIGIN_FOG_4H',
#  'od_median_delay_2H',
#  'od_median_delay_4H',
#  'delay_state_2017_2H',
#  'delay_state_2017_4H',
#  'outbound_delay_2H',
#  'outbound_delay_4H',
#  'inbound_delay_2H',
#  'inbound_delay_4H',
#  'DEP_DEL15']

# COMMAND ----------

include = ['utc2H',
'OP_CARRIER_FL_NUM',
'OP_UNIQUE_CARRIER',
'YEAR',
'MONTH',
'DAY_OF_WEEK',
'QUARTER',
'DAY_OF_MONTH',
'DISTANCE_GROUP',
'ORIGIN_TS_2H',
'ORIGIN_ICE_2H',
'ORIGIN_SNOW_2H',
'ORIGIN_FOG_2H',
'DEST_TS_2H',
'DEST_ICE_2H',
'DEST_SNOW_2H',
'DEST_FOG_2H',
'ORIGIN_TS_4H',
'ORIGIN_ICE_4H',
'ORIGIN_SNOW_4H',
'ORIGIN_FOG_4H',
'DEST_TS_4H',
'DEST_ICE_4H',
'DEST_SNOW_4H',
'DEST_FOG_4H',
'od_median_delay_2H',
'delay_state_2018_2H',
'delay_state_2017_2H',
'delay_state_2016_2H',
'delay_state_2015_2H',
'outbound_delay_2H',
'inbound_delay_2H',
'od_median_delay_4H',
'delay_state_2018_4H',
'delay_state_2017_4H',
'delay_state_2016_4H',
'delay_state_2015_4H',
'outbound_delay_4H',
'inbound_delay_4H',
'origin_pagerank',
'dest_pagerank',
'prev_dep_delay_fair_1',
'mean_prev_dep_delay',
'DEP_DEL15']



# COMMAND ----------

# load omni file
df = spark.read.parquet(f'{blob_url}/omni_file/')

# filter out cancelled and diverted flights
df = df.filter((F.col('CANCELLED') == 0) & (F.col('DIVERTED') == 0))


# COMMAND ----------

# filter to only include needed columns
# replace null values with 0
df = df[include].na.fill(0).cache()
display(df)

# COMMAND ----------

# convert delay features to binary
df = df.withColumn('od_median_delay_2H_binary', F.when(df.od_median_delay_2H > 15, 1).otherwise(0))\
       .withColumn('od_median_delay_4H_binary', F.when(df.od_median_delay_4H > 15, 1).otherwise(0))\
       .withColumn('outbound_delay_2H_binary', F.when(df.outbound_delay_2H > 15, 1).otherwise(0))\
       .withColumn('outbound_delay_4H_binary', F.when(df.outbound_delay_4H > 15, 1).otherwise(0))\
       .withColumn('inbound_delay_2H_binary', F.when(df.inbound_delay_2H > 15, 1).otherwise(0))\
       .withColumn('inbound_delay_4H_binary', F.when(df.inbound_delay_4H > 15, 1).otherwise(0))\
       .cache()

# COMMAND ----------

# columns to be used as features
feature_cols = ['MONTH',
 'DAY_OF_WEEK',
 'YEAR',
 'DEST_TS_2H',
 'DEST_ICE_2H',
 'DEST_SNOW_2H',
 'DEST_FOG_2H',
 'ORIGIN_TS_2H',
 'ORIGIN_ICE_2H',
 'ORIGIN_SNOW_2H',
 'ORIGIN_FOG_2H',
'od_median_delay_2H',
'delay_state_2018_2H',
'delay_state_2017_2H',
'delay_state_2016_2H',
'delay_state_2015_2H',
'outbound_delay_2H',
'inbound_delay_2H',
'origin_pagerank',
'dest_pagerank',
'prev_dep_delay_fair_1',
'mean_prev_dep_delay'
]

# COMMAND ----------

# how many features are we keeping?
len(feature_cols)

# COMMAND ----------

# create input vectors for modeling
assemble = VectorAssembler(outputCol='features')
assemble.setInputCols(feature_cols)
assembled_data = assemble.transform(df)
#display(assembled_data)

# COMMAND ----------

# keep only features, labels, and columns for filtering
# rf_input = assembled_data[['utc', 'OP_CARRIER_FL_NUM', 'OP_UNIQUE_CARRIER', 'DEP_DEL15', 'features']]
rf_input = assembled_data[['utc2H', 'DEP_DEL15', 'features']]
rf_input = rf_input.withColumnRenamed('DEP_DEL15', 'label')

# COMMAND ----------

# how imbalanced is our train data?
counts = rf_input.filter(F.year(rf_input.utc2H) <= 2017).select('label').groupBy('label').count().toPandas()
display(counts)

# COMMAND ----------

# train data counts
delayed_count = counts['count'][1]
total_count = counts['count'].sum()

# weights
c = 2
weight_delayed = total_count / (c * delayed_count)
weight_not_delayed = total_count / (c * (total_count - delayed_count))

# append weights to dataset
rf_weighted = rf_input.withColumn('weight', F.when(F.col('label') == 1, weight_delayed).otherwise(weight_not_delayed))

# COMMAND ----------

train = rf_weighted.filter((F.year(rf_weighted.utc2H) <= 2017))

# COMMAND ----------

train = train[['label', 'features', 'weight']]

# COMMAND ----------

display(train)
train.count()

# COMMAND ----------

rf = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="weight", numTrees=50, predictionCol='rf_pred', probabilityCol='rf_prob', rawPredictionCol='rf_raw')
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxBins=350, predictionCol='dt_pred', probabilityCol='dt_prob', rawPredictionCol='dt_raw')

# COMMAND ----------

#Create RF and DT baseline models and use it to train data
gen_base_pred_pipeline = Pipeline(stages=[rf, dt])
gen_base_pred_pipeline_model = gen_base_pred_pipeline.fit(train)

# COMMAND ----------

#Using the baseline model predict on the same training set
baseline_preds = gen_base_pred_pipeline_model.transform(train)
display(baseline_preds)

# COMMAND ----------

#Drop the flight carrier, utc, features columns (if there are more than 1 label column - drop that too)
baseline_preds = baseline_preds[['label', 'weight', 'rf_pred', 'dt_pred']]
display(baseline_preds)

# COMMAND ----------

#vectorize the prediction columns into features
vector_assembler = VectorAssembler(inputCols=['rf_pred', 'dt_pred'], outputCol='features')

#Create pipeline and pass it to stages
pipeline = Pipeline(stages=[vector_assembler])
#fit and transform
transformed_baseline_preds = pipeline.fit(baseline_preds).transform(baseline_preds)
display(transformed_baseline_preds)


# COMMAND ----------

transformed_baseline_preds = transformed_baseline_preds[['label', 'weight', 'features']]
display(transformed_baseline_preds)

# COMMAND ----------

# Set up logistic regression model - final level in stacking
lr_model = LogisticRegression(featuresCol='features', labelCol='label', predictionCol='final_prediction', maxIter=100)
#training model
meta_classifier = lr_model.fit(transformed_baseline_preds)

# COMMAND ----------

# create test set
test = rf_weighted.filter(F.year(rf_weighted.utc2H) == 2018)
test = test[['label', 'features', 'weight']]

# COMMAND ----------

display(test)
test.count()

# COMMAND ----------

#Using the baseline model predict on the same training set
test_baseline_preds = gen_base_pred_pipeline_model.transform(test)
display(test_baseline_preds)

# COMMAND ----------

#Drop the flight carrier, utc, features columns (if there are more than 1 label column - drop that too)
test_baseline_preds = test_baseline_preds[['label', 'weight', 'rf_pred', 'dt_pred']]
display(test_baseline_preds)

# COMMAND ----------

#vectorize the prediction columns into features
vector_assembler = VectorAssembler(inputCols=['rf_pred', 'dt_pred'], outputCol='features')

#Create pipeline and pass it to stages
pipeline = Pipeline(stages=[vector_assembler])
#fit and transform
test_transformed_baseline_preds = pipeline.fit(test_baseline_preds).transform(test_baseline_preds)
display(test_transformed_baseline_preds)

# COMMAND ----------

test_transformed_baseline_preds = test_transformed_baseline_preds[['label', 'weight', 'features']]
display(test_transformed_baseline_preds)

# COMMAND ----------

# prediction on test data
preds = meta_classifier.transform(test_transformed_baseline_preds)

# COMMAND ----------

display(preds)
display(preds.groupby('label', 'final_prediction').count())

# COMMAND ----------

####Done baseline modeling of stacking####
####New stacking model with multiple RF in layer 1####

# COMMAND ----------

rf_01 = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="weight", numTrees=50, predictionCol='rf_pred_01', probabilityCol='rf_prob_01', rawPredictionCol='rf_raw_01')
rf_02 = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="weight", numTrees=50, predictionCol='rf_pred_02', probabilityCol='rf_prob_02', rawPredictionCol='rf_raw_02')
rf_03 = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="weight", numTrees=50, predictionCol='rf_pred_03', probabilityCol='rf_prob_03', rawPredictionCol='rf_raw_03')
gbt = GBTClassifier(labelCol="label", featuresCol="features", weightCol="weight", predictionCol='gbt_pred', maxIter=50)
# dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxBins=350, predictionCol='dt_pred', probabilityCol='dt_prob', rawPredictionCol='dt_raw')

# COMMAND ----------

#Create RF, and GBT baseline models and use it to train data
gen_base_pred_pipeline = Pipeline(stages=[rf_01, rf_02, gbt])
gen_base_pred_pipeline_model = gen_base_pred_pipeline.fit(train)

# COMMAND ----------

#Using the baseline model predict on the same training set
baseline_preds = gen_base_pred_pipeline_model.transform(train)
display(baseline_preds)

# COMMAND ----------

#Drop the flight carrier, utc, features columns (if there are more than 1 label column - drop that too)
baseline_preds = baseline_preds[['label', 'weight', 'rf_pred_01', 'rf_pred_02', 'gbt_pred']]
display(baseline_preds)

# COMMAND ----------

#vectorize the prediction columns into features
vector_assembler = VectorAssembler(inputCols=['rf_pred_01', 'rf_pred_02', 'gbt_pred'], outputCol='features')

#Create pipeline and pass it to stages
pipeline = Pipeline(stages=[vector_assembler])
#fit and transform
transformed_baseline_preds = pipeline.fit(baseline_preds).transform(baseline_preds)
display(transformed_baseline_preds)

# COMMAND ----------

transformed_baseline_preds = transformed_baseline_preds[['label', 'weight', 'features']]
display(transformed_baseline_preds)

# COMMAND ----------

# Set up logistic regression model - final level in stacking
lr_model = LogisticRegression(featuresCol='features', labelCol='label', predictionCol='final_prediction', maxIter=100)
#training model
meta_classifier = lr_model.fit(transformed_baseline_preds)

# COMMAND ----------

# create test set
test = rf_weighted.filter(F.year(rf_weighted.utc2H) == 2018)
test = test[['label', 'features', 'weight']]

# COMMAND ----------

#Using the baseline model predict on the same training set
test_baseline_preds = gen_base_pred_pipeline_model.transform(test)
display(test_baseline_preds)

# COMMAND ----------

#Drop the flight carrier, utc, features columns (if there are more than 1 label column - drop that too)
test_baseline_preds = test_baseline_preds[['label', 'weight', 'rf_pred_01', 'rf_pred_02', 'gbt_pred']]
display(test_baseline_preds)

# COMMAND ----------

display(test_baseline_preds.groupby('label', 'rf_pred_01', 'rf_pred_02', 'gbt_pred').count())

# COMMAND ----------

#vectorize the prediction columns into features
vector_assembler = VectorAssembler(inputCols=['rf_pred_01', 'rf_pred_02', 'gbt_pred'], outputCol='features')

#Create pipeline and pass it to stages
pipeline = Pipeline(stages=[vector_assembler])
#fit and transform
test_transformed_baseline_preds = pipeline.fit(test_baseline_preds).transform(test_baseline_preds)
display(test_transformed_baseline_preds)

# COMMAND ----------

test_transformed_baseline_preds = test_transformed_baseline_preds[['label', 'weight', 'features']]
display(test_transformed_baseline_preds)

# COMMAND ----------

# prediction on test data
preds = meta_classifier.transform(test_transformed_baseline_preds)

# COMMAND ----------

display(preds)
display(preds.groupby('label', 'final_prediction').count())

# COMMAND ----------

# #Evaluate model
# evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName='f1')
# print(evaluator.evaluate(preds, {evaluator.predictionCol:'final_prediction'}))

# COMMAND ----------

#####Seperate features for each model####

# COMMAND ----------

rf_01 = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="weight", numTrees=50, predictionCol='rf_pred_01', probabilityCol='rf_prob_01', rawPredictionCol='rf_raw_01')
rf_02 = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="weight", numTrees=50, predictionCol='rf_pred_02', probabilityCol='rf_prob_02', rawPredictionCol='rf_raw_02')
rf_03 = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="weight", numTrees=50, predictionCol='rf_pred_03', probabilityCol='rf_prob_03', rawPredictionCol='rf_raw_03')
gbt = GBTClassifier(labelCol="label", featuresCol="features", weightCol="weight", predictionCol='gbt_pred', maxIter=50)

# COMMAND ----------


