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
from pyspark.ml.linalg import DenseVector, SparseVector, Vectors

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

# Load the parquet with all usable features for our model
omni = spark.read.parquet(f"{blob_url}/omni_file")

# COMMAND ----------

# Definition of Features

target_features = ['DEP_DEL15', 
                   'DEP_DELAY']

original_features = ['YEAR',
                    'MONTH',
                    'DAY_OF_WEEK'
                    'QUARTER',
                    'DAY_OF_MONTH',
                    'DISTANCE_GROUP',
                    'OP_CARRIER_AIRLINE_ID',
                    'ORIGIN_TS_2H',
                    'ORIGIN_ICE_2H',
                    'ORIGIN_SNOW_2H',
                    'ORIGIN_FOG_2H',
                    'DEST_TS_2H',
                    'DEST_ICE_2H',
                    'DEST_SNOW_2H',
                    'DEST_FOG_2H'
                    'delay_state_2018_2H',
                    'delay_state_2017_2H',
                    'delay_state_2016_2H',
                    'delay_state_2015_2H',
                    'delay_state_2018_4H',
                    'delay_state_2017_4H',
                    'delay_state_2016_4H',
                    'delay_state_2015_4H',
                    'prev_dep_delay_fair_1',
                    'prev_dep_delay_fair_2',
                    'prev_dep_delay_fair_3',
                    'prev_dep_delay_fair_4',
                    'prev_dep_delay_fair_5',
                    'prev_arr_delay_fair_1',
                    'prev_arr_delay_fair_2',
                    'prev_arr_delay_fair_3',
                    'prev_arr_delay_fair_4',
                    'prev_arr_delay_fair_5']


replace_minus_1 = ['ORIGIN_TS_2H',
'ORIGIN_ICE_2H',
'ORIGIN_SNOW_2H',
'ORIGIN_FOG_2H',
'DEST_TS_2H',
'DEST_ICE_2H',
'DEST_SNOW_2H',
'DEST_FOG_2H',
'delay_state_2018_2H',
'delay_state_2017_2H',
'delay_state_2016_2H',
'delay_state_2015_2H',
'delay_state_2018_4H',
'delay_state_2017_4H',
'delay_state_2016_4H',
'delay_state_2015_4H']

features = ['YEAR', #1
                    'MONTH', #2
                    'DAY_OF_WEEK', #3
                    'QUARTER',#4
                    'DAY_OF_MONTH', #5
                    'DISTANCE_GROUP', #6
                    'OP_CARRIER_AIRLINE_ID', #7
                    'ORIGIN_TS_2H', #8
                    'ORIGIN_ICE_2H', #9
                    'ORIGIN_SNOW_2H', #10
                    'ORIGIN_FOG_2H', #11
                    'DEST_TS_2H', #12
                    'DEST_ICE_2H', #13
                    'DEST_SNOW_2H', #14
                    'DEST_FOG_2H', #15
                    'delay_state_2018_2H', #16
                    'delay_state_2017_2H', #17
                    'delay_state_2016_2H', #18
                    'delay_state_2015_2H', #19
                    'delay_state_2018_4H', #20
                    'delay_state_2017_4H', #21
                    'delay_state_2016_4H', #22
                    'delay_state_2015_4H', #23
                    'prev_dep_delay_fair_1_binned', #24
                    'prev_dep_delay_fair_2_binned', #25
                    'prev_dep_delay_fair_3_binned', #26
                    'prev_dep_delay_fair_4_binned', #27
                    'prev_dep_delay_fair_5_binned', #28
                    'prev_arr_delay_fair_1_binned', #29
                    'prev_arr_delay_fair_2_binned', #30
                    'prev_arr_delay_fair_3_binned', #31
                    'prev_arr_delay_fair_4_binned', #32
                    'prev_arr_delay_fair_5_binned'] #33

featuresNoWeather = ['YEAR', #1
                    'MONTH', #2
                    'DAY_OF_WEEK', #3
                    'QUARTER',#4
                    'DAY_OF_MONTH', #5
                    'DISTANCE_GROUP', #6
                    'OP_CARRIER_AIRLINE_ID', #7
                    'delay_state_2018_2H', #8
                    'delay_state_2017_2H', #9
                    'delay_state_2016_2H', #10
                    'delay_state_2015_2H', #11
                    'delay_state_2018_4H', #12
                    'delay_state_2017_4H', #13
                    'delay_state_2016_4H', #14
                    'delay_state_2015_4H', #15
                    'prev_dep_delay_fair_1_binned', #16
                    'prev_dep_delay_fair_2_binned', #17
                    'prev_dep_delay_fair_3_binned', #18
                    'prev_dep_delay_fair_4_binned', #19
                    'prev_dep_delay_fair_5_binned', #20
                    'prev_arr_delay_fair_1_binned', #21
                    'prev_arr_delay_fair_2_binned', #22
                    'prev_arr_delay_fair_3_binned', #23
                    'prev_arr_delay_fair_4_binned', #24
                    'prev_arr_delay_fair_5_binned'] #25

# COMMAND ----------

# Handle Target and Weather nulls: Drop rows with nulls for target features, and replace nulls with -1

omni_2 = omni.na.drop(subset=["DEP_DEL15"]).fillna(-1, subset=replace_minus_1) \
             .withColumn('prev_dep_delay_fair_1_binned', F.when((F.col('prev_dep_delay_fair_1') < 0), 0) \
                                        .when((F.col('prev_dep_delay_fair_1') >= 0) & (F.col('prev_dep_delay_fair_1') < 15), 1) \
                                        .when((F.col('prev_dep_delay_fair_1') >= 15) & (F.col('prev_dep_delay_fair_1') < 30), 2) \
                                        .when((F.col('prev_dep_delay_fair_1') >= 30) & (F.col('prev_dep_delay_fair_1') < 44), 3) \
                                        .when((F.col('prev_dep_delay_fair_1') >= 45), 4).otherwise(-1)) \
             .withColumn('prev_dep_delay_fair_2_binned', F.when((F.col('prev_dep_delay_fair_2') < 0), 0) \
                                        .when((F.col('prev_dep_delay_fair_2') >= 0) & (F.col('prev_dep_delay_fair_2') < 15), 1) \
                                        .when((F.col('prev_dep_delay_fair_2') >= 15) & (F.col('prev_dep_delay_fair_2') < 30), 2) \
                                        .when((F.col('prev_dep_delay_fair_2') >= 30) & (F.col('prev_dep_delay_fair_2') < 44), 3) \
                                        .when((F.col('prev_dep_delay_fair_2') >= 45), 4).otherwise(-1)) \
             .withColumn('prev_dep_delay_fair_3_binned', F.when((F.col('prev_dep_delay_fair_3') < 0), 0) \
                                        .when((F.col('prev_dep_delay_fair_3') >= 0) & (F.col('prev_dep_delay_fair_3') < 15), 1) \
                                        .when((F.col('prev_dep_delay_fair_3') >= 15) & (F.col('prev_dep_delay_fair_3') < 30), 2) \
                                        .when((F.col('prev_dep_delay_fair_3') >= 30) & (F.col('prev_dep_delay_fair_3') < 44), 3) \
                                        .when((F.col('prev_dep_delay_fair_3') >= 45), 4).otherwise(-1)) \
             .withColumn('prev_dep_delay_fair_4_binned', F.when((F.col('prev_dep_delay_fair_4') < 0), 0) \
                                        .when((F.col('prev_dep_delay_fair_4') >= 0) & (F.col('prev_dep_delay_fair_4') < 15), 1) \
                                        .when((F.col('prev_dep_delay_fair_4') >= 15) & (F.col('prev_dep_delay_fair_4') < 30), 2) \
                                        .when((F.col('prev_dep_delay_fair_4') >= 30) & (F.col('prev_dep_delay_fair_4') < 44), 3) \
                                        .when((F.col('prev_dep_delay_fair_4') >= 45), 4).otherwise(-1)) \
             .withColumn('prev_dep_delay_fair_5_binned', F.when((F.col('prev_dep_delay_fair_5') < 0), 0) \
                                        .when((F.col('prev_dep_delay_fair_5') >= 0) & (F.col('prev_dep_delay_fair_5') < 15), 1) \
                                        .when((F.col('prev_dep_delay_fair_5') >= 15) & (F.col('prev_dep_delay_fair_5') < 30), 2) \
                                        .when((F.col('prev_dep_delay_fair_5') >= 30) & (F.col('prev_dep_delay_fair_5') < 44), 3) \
                                        .when((F.col('prev_dep_delay_fair_5') >= 45), 4).otherwise(-1)) \
             .withColumn('prev_arr_delay_fair_1_binned', F.when((F.col('prev_arr_delay_fair_1') < 0), 0) \
                                        .when((F.col('prev_arr_delay_fair_1') >= 0) & (F.col('prev_arr_delay_fair_1') < 15), 1) \
                                        .when((F.col('prev_arr_delay_fair_1') >= 15) & (F.col('prev_arr_delay_fair_1') < 30), 2) \
                                        .when((F.col('prev_arr_delay_fair_1') >= 30) & (F.col('prev_arr_delay_fair_1') < 44), 3) \
                                        .when((F.col('prev_arr_delay_fair_1') >= 45), 4).otherwise(-1)) \
             .withColumn('prev_arr_delay_fair_2_binned', F.when((F.col('prev_arr_delay_fair_2') < 0), 0) \
                                        .when((F.col('prev_arr_delay_fair_2') >= 0) & (F.col('prev_arr_delay_fair_2') < 15), 1) \
                                        .when((F.col('prev_arr_delay_fair_2') >= 15) & (F.col('prev_arr_delay_fair_2') < 30), 2) \
                                        .when((F.col('prev_arr_delay_fair_2') >= 30) & (F.col('prev_arr_delay_fair_2') < 44), 3) \
                                        .when((F.col('prev_arr_delay_fair_2') >= 45), 4).otherwise(-1)) \
             .withColumn('prev_arr_delay_fair_3_binned', F.when((F.col('prev_arr_delay_fair_3') < 0), 0) \
                                        .when((F.col('prev_arr_delay_fair_3') >= 0) & (F.col('prev_arr_delay_fair_3') < 15), 1) \
                                        .when((F.col('prev_arr_delay_fair_3') >= 15) & (F.col('prev_arr_delay_fair_3') < 30), 2) \
                                        .when((F.col('prev_arr_delay_fair_3') >= 30) & (F.col('prev_arr_delay_fair_3') < 44), 3) \
                                        .when((F.col('prev_arr_delay_fair_3') >= 45), 4).otherwise(-1)) \
             .withColumn('prev_arr_delay_fair_4_binned', F.when((F.col('prev_arr_delay_fair_4') < 0), 0) \
                                        .when((F.col('prev_arr_delay_fair_4') >= 0) & (F.col('prev_arr_delay_fair_4') < 15), 1) \
                                        .when((F.col('prev_arr_delay_fair_4') >= 15) & (F.col('prev_arr_delay_fair_4') < 30), 2) \
                                        .when((F.col('prev_arr_delay_fair_4') >= 30) & (F.col('prev_arr_delay_fair_4') < 44), 3) \
                                        .when((F.col('prev_arr_delay_fair_4') >= 45), 4).otherwise(-1)) \
             .withColumn('prev_arr_delay_fair_5_binned', F.when((F.col('prev_arr_delay_fair_5') < 0), 0) \
                                        .when((F.col('prev_arr_delay_fair_5') >= 0) & (F.col('prev_arr_delay_fair_5') < 15), 1) \
                                        .when((F.col('prev_arr_delay_fair_5') >= 15) & (F.col('prev_arr_delay_fair_5') < 30), 2) \
                                        .when((F.col('prev_arr_delay_fair_5') >= 30) & (F.col('prev_arr_delay_fair_5') < 44), 3) \
                                        .when((F.col('prev_arr_delay_fair_5') >= 45), 4).otherwise(-1))

# COMMAND ----------

display(omni_2)

# COMMAND ----------

data = omni_2.select('DEP_DEL15', *features)
display(data)
display(data.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in data.columns]))

# COMMAND ----------

def train_predict_dt(df, features, target, bins, depth):
  """Helper function to train and test a decision tree with different dephts. Harcoded train and test period (2015 and Q1 2016)"""
  
  # Define the data to be used
  train, test = df.filter(df.YEAR <= 2017), df.filter(df.YEAR == 2018)
  
  # make vector with all features, except the target variable
  vector_assembler = VectorAssembler(inputCols=features, outputCol="features_vector")
  trainingData, testData = vector_assembler.setHandleInvalid("keep").transform(train), vector_assembler.setHandleInvalid("keep").transform(test)

  # Train a DecisionTree model.
  dt = DecisionTreeClassifier(labelCol=target, featuresCol="features_vector", maxBins=bins, maxDepth=depth)

  # Chain indexers and tree in a Pipeline
  # pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
  pipeline = Pipeline(stages=[dt])

  # Train model.  This also runs the indexers.
  model = pipeline.fit(trainingData)

  # Make predictions.
  predictions = model.transform(testData)
  
  return model, predictions

# COMMAND ----------

model_5, predictions_5 = train_predict_dt(data, features, 'DEP_DEL15', 350, 5)
model_10, predictions_10 = train_predict_dt(data, features, 'DEP_DEL15', 350, 10)
model_15, predictions_15 = train_predict_dt(data, features, 'DEP_DEL15', 350, 15)
model_20, predictions_20 = train_predict_dt(data, features, 'DEP_DEL15', 350, 20)

# COMMAND ----------

print("Results for dept = 5")
display(predictions_5.groupby('DEP_DEL15', 'prediction').count())
display(model_5.stages[0])

# COMMAND ----------

print("Results for dept = 10")
display(predictions_10.groupby('DEP_DEL15', 'prediction').count())
display(model_10.stages[0])

# COMMAND ----------

print("Results for dept = 15")
display(predictions_15.groupby('DEP_DEL15', 'prediction').count())
display(model_15.stages[0])

# COMMAND ----------

print("Results for dept = 20")
display(predictions_20.groupby('DEP_DEL15', 'prediction').count())
display(model_20.stages[0])

# COMMAND ----------

model_noWeather_5, predictions_noWeather_5 = train_predict_dt(data, featuresNoWeather, 'DEP_DEL15', 350, 5)
model_noWeather_10, predictions_noWeather_10 = train_predict_dt(data, featuresNoWeather, 'DEP_DEL15', 350, 10)
model_noWeather_15, predictions_noWeather_15 = train_predict_dt(data, featuresNoWeather, 'DEP_DEL15', 350, 15)
model_noWeather_20, predictions_noWeather_20 = train_predict_dt(data, featuresNoWeather, 'DEP_DEL15', 350, 20)

# COMMAND ----------

print("Results for dept = 5 - No Weather")
display(predictions_noWeather_5.groupby('DEP_DEL15', 'prediction').count())
display(model_noWeather_5.stages[0])

# COMMAND ----------

print("Results for dept = 10 - No Weather")
display(predictions_noWeather_10.groupby('DEP_DEL15', 'prediction').count())
display(model_noWeather_10.stages[0])

# COMMAND ----------

print("Results for dept = 15 - No Weather")
display(predictions_noWeather_15.groupby('DEP_DEL15', 'prediction').count())
display(model_noWeather_15.stages[0])

# COMMAND ----------

print("Results for dept = 20 - No Weather")
display(predictions_noWeather_20.groupby('DEP_DEL15', 'prediction').count())
display(model_noWeather_20.stages[0])

# COMMAND ----------

# # Separate string features from the rest to use the indexer

# def get_feature_types(df):
#   """Helper function to get names of all features that are strings and not strings"""
#   features = []
#   stringFeatures = []

#   for pair in df.dtypes: 
#     if pair[1] == 'string':
#       stringFeatures.append(pair[0])
#     else:
#       features.append(pair[0])
      
#   return features, stringFeatures

# COMMAND ----------

# variables, stringVariables = get_feature_types(data)

# COMMAND ----------

# variables

# COMMAND ----------

# stringVariables

# COMMAND ----------

# Index cathegorical string features
# # use StringIndexer and Pipeline to index all string features

# def index_features(df, features):
#   """Helper function to index variables passed as a list"""

#   indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in features ]
#   pipeline = Pipeline(stages=indexers)
#   df_indexed = pipeline.fit(df).transform(df)

#   return df_indexed

# COMMAND ----------

# df_indexed = index_features(data, stringFeatures)

# COMMAND ----------

# display(df_indexed)

# COMMAND ----------

# # Decision tree for classification

# # Make vector with usable features
# vector_assembler = VectorAssembler(inputCols=features, outputCol="features_vector")
# df_vector = vector_assembler.setHandleInvalid("keep").transform(df_indexed)
# display(df_indexed_vector)

# COMMAND ----------

# # Split train and test
# train = df_vector.filter(df_vector.YEAR <= 2017)
# test = df_vector.filter(df_vector.YEAR == 2018)

# COMMAND ----------


# # make vector with all features, except the target variable

# vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
# data_vector = vector_assembler.setHandleInvalid("keep").transform(data)
# display(data_vector)

# # Index labels, adding metadata to the label column.
# # Fit on whole dataset to include all labels in index.
# #labelIndexer = StringIndexer(inputCol="DEP_DEL15", outputCol="indexedLabel").fit(data_vector)
# # Automatically identify categorical features, and index them.
# # We specify maxCategories so features with > 20 distinct values are treated as continuous.
# #featureIndexer =\
# #    VectorIndexer(inputCol="features_vector", outputCol="indexedFeatures", maxCategories=20).fit(data_vector)

# # Split the data into training and test sets (30% held out for testing)
# trainingData, testData = data_vector.filter(data_vector.YEAR <= 2017), data_vector.filter(data_vector.YEAR == 2018)

# # Train a DecisionTree model.
# dt = DecisionTreeClassifier(labelCol="DEP_DEL15", featuresCol="features", maxBins=350)

# # Chain indexers and tree in a Pipeline
# # pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
# pipeline = Pipeline(stages=[dt])

# # Train model.  This also runs the indexers.
# model = pipeline.fit(trainingData)

# # Make predictions.
# predictions = model.transform(testData)

# # Select example rows to display.
# predictions.select("prediction", "DEP_DEL15", "features").show(5)

# COMMAND ----------

# treeModel = model.stages[0]
# # # summary only
# print(treeModel)
# display(treeModel)
