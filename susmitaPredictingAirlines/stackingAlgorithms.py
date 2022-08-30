# Databricks notebook source
import numpy as np 
from sklearn import tree
import matplotlib.pyplot as plt

from pyspark.mllib.tree import DecisionTree

import pyspark.sql.functions as sf

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, VectorSlicer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.util import MLUtils
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# Azure blob storage init script
blob_container = "container1" # The name of your container created in https://portal.azure.com
storage_account = "w261f2021s1t1" # The name of your Storage account created in https://portal.azure.com
secret_scope = "scope1" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "key1" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key))

# COMMAND ----------

display(dbutils.fs.ls("wasbs://container1@w261f2021s1t1.blob.core.windows.net/"))

# COMMAND ----------

df_joined_full = spark.read.parquet("wasbs://container1@w261f2021s1t1.blob.core.windows.net/parquet_flight_weather_network_joined/")

# COMMAND ----------

display(df_joined_full)

# COMMAND ----------

dfj_201518 = df_joined_full.filter(df_joined_full.YEAR <= 2018).na.drop(subset=["DEP_TIME","DEP_DELAY", "ARR_TIME", "ARR_DELAY"]).fillna(0).fillna('0')

# COMMAND ----------

display(dfj_201518)

# COMMAND ----------

# separate string features from the rest to use the indexer

def get_feature_types(df):
  """Function to get names of all features that are strings and not strings"""
  features = []
  stringFeatures = []

  for pair in df.dtypes: 
    if pair[1] == 'string':
      stringFeatures.append(pair[0])
    else:
      features.append(pair[0])
      
  return features, stringFeatures

# COMMAND ----------

# Index cathegorical string features
# use StringIndexer and Pipeline to index all string features

def index_features(df, features):
  """Helper function to index variables passed as a list"""

  indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in features ]
  pipeline = Pipeline(stages=indexers)
  df_indexed = pipeline.fit(df).transform(df)

  return df_indexed

# COMMAND ----------

variables, stringVariables = get_feature_types(df_joined_full)

# COMMAND ----------

df_indexed = spark.read.parquet("wasbs://container1@w261f2021s1t1.blob.core.windows.net/indexed_usable_features_2015_2018.parquet/")

# COMMAND ----------

display(df_indexed)

# COMMAND ----------

features, stringFeatures = get_feature_types(df_indexed)

# COMMAND ----------

features = list(set(features) - set(target_features))

# COMMAND ----------

time_features = ['utc', 'utc_hour_trunc', 'utc_dep_date']
features = list(set(features)-set(time_features))

# COMMAND ----------

weather_features = ['DEST_TS_2H', 'DEST_ICE_2H', 'DEST_SNOW_2H', 'DEST_FOG_2H', 'DEST_TS_4H', 'DEST_ICE_4H', 'DEST_SNOW_4H', 'DEST_FOG_4H', 'ORIGIN_TS_2H', 'ORIGIN_ICE_2H', 'ORIGIN_SNOW_2H', 'ORIGIN_FOG_2H', 'ORIGIN_TS_4H', 'ORIGIN_ICE_4H', 'ORIGIN_SNOW_4H', 'ORIGIN_FOG_4H']
features = list(set(features)-set(weather_features))

# COMMAND ----------

df_indexed.drop(*time_features)
df_indexed.drop(*weather_features)

# COMMAND ----------

all_columns = features + target_features

# COMMAND ----------

display(df_indexed.select(all_columns))

# COMMAND ----------

display(df_indexed)

# COMMAND ----------

features

# COMMAND ----------

target_features

# COMMAND ----------

# transformer
vector_assembler = VectorAssembler(inputCols=features,outputCol="features")
df_temp = vector_assembler.setHandleInvalid("keep").transform(df_indexed)
display(df_temp)

# data splitting
train = df_temp.filter(df_temp.YEAR <= 2017)
test = df_temp.filter(df_temp.YEAR == 2018)

# drop the original data features column
train = train.drop(*features).drop(*time_features).drop(*stringFeatures)
test = test.drop(*features).drop(*time_features).drop(*stringFeatures)
display(train)
display(test)

# COMMAND ----------

# one hot encoding and assembling
encoding_var = [i[0] for i in df_indexed.dtypes if (i[1]=='string') & (i[0]!='y')]
num_var = [i[0] for i in df_indexed.dtypes if ((i[1]=='int') | (i[1]=='double')) & (i[0]!='y')]

'''from string to interger'''
string_indexes = [StringIndexer(inputCol = c, outputCol = 'IDX_' + c, handleInvalid = 'keep') for c in encoding_var]
'''from interger to binary vectors'''
onehot_indexes = [OneHotEncoder(inputCols = ['IDX_' + c], outputCols = ['OHE_' + c]) for c in encoding_var]
label_indexes = StringIndexer(inputCol = 'y', outputCol = 'label', handleInvalid = 'keep')


## The input for the model should be binary vectors
assembler = VectorAssembler(inputCols = num_var + ['OHE_' + c for c in encoding_var], outputCol = "features")

# COMMAND ----------

vector_assembler = VectorAssembler(inputCols=features, outputCol="features_vector")
df_vector = vector_assembler.setHandleInvalid("keep").transform(df_indexed)
display(df_vector)

# COMMAND ----------

train = df_vector.filter(df_vector.YEAR <= 2017)
test = df_vector.filter(df_vector.YEAR == 2018)

# COMMAND ----------

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier, GBTClassifier,LogisticRegression, NaiveBayes

rf = RandomForestClassifier(numTrees=20)
xgb = GBTClassifier(maxIter= 10)
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.2)
#nb = NaiveBayes(smoothing= 0.5, modelType="multinomial")

methods = {"random forest": rf,
           "logistic regression": lr,
          "boosting tree": xgb ##this needs to be different from others
          #"naive bayes": nb
          }

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
fitted_models ={}

for method_name, method in methods.items():
    
    method.setPredictionCol("prediction_" + method_name)
    if method_name != "boosting tree":
        method.setProbabilityCol("probability_" + method_name)
        method.setRawPredictionCol("raw_prediction_" + method_name)
        sel_col = "probability_" + method_name
    else:
        sel_col = "probability"
    

    pipe = Pipeline(stages = string_indexes + onehot_indexes + [assembler, label_indexes, method])
    # need to keep fitted model somewhere
    fitted_models[method_name] = pipe.fit(train)
    df_test = fitted_models[method_name].transform(test)
    
    filter_col1 = [col for col in df_test.columns if col.startswith('IDX')]
    filter_col2 = [col for col in df_test.columns if col.startswith('OHE')]
    drop_cols = filter_col1 + filter_col2 + ['features','label']
#     drop_cols = features
    
    evaluator= BinaryClassificationEvaluator(rawPredictionCol=sel_col, metricName= "areaUnderROC")
    print(evaluator.evaluate(df_test))
    if method_name != list(methods.keys())[len(methods)-1]: ##if it is the last layer, we will not drop columns
        df_test = df_test.drop(*drop_cols)

# COMMAND ----------

display(df_test)

# COMMAND ----------


