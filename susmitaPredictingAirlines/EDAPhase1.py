# Databricks notebook source
from pyspark.sql.functions import col
from graphframes import *
from pyspark.sql.functions import col,when,count
from pyspark.sql import functions as sf

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

df_airlines3 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines3)

# COMMAND ----------

df_airlines6 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_6m/")
display(df_airlines6)

# COMMAND ----------

# 1) Explain data (i.e.., simple exploratory analysis of various fields, such as the semantic as well as intrinsic meaning of ranges, null values, categorical/numerical, mean/std.dev to normalize and/or scale inputs). Identify any missing or corrupt (i.e., outlier) data.
# The main fields to pay attention to in the dataset is year and quarter to distinguish temporal data. The origin airport ID, destination airport ID, arrival delay, destination delay, desitnation or arrival delay more than 15 mins flag

# 2) Define the outcome (i.e., the evaluation metric and the target) precisely, including mathematical formulas.


# COMMAND ----------

# Stats on certain columns; describe data
#df_airlines3["ARR_DELAY"].describe()

# Range of values for types of delays
df_airlines3.agg({'ARR_DELAY': 'avg',
               'DEP_DELAY': 'avg',
                'CARRIER_DELAY': 'avg',
                 'WEATHER_DELAY': 'avg',
                 'NAS_DELAY': 'avg',
                 'SECURITY_DELAY': 'avg',
                 'LATE_AIRCRAFT_DELAY': 'avg'}).show()

df_airlines3.agg({'ARR_DELAY': 'max',
               'DEP_DELAY': 'max',  
                  'CARRIER_DELAY': 'max',
                 'WEATHER_DELAY': 'max',
                 'NAS_DELAY': 'max',
                 'SECURITY_DELAY': 'max',
                 'LATE_AIRCRAFT_DELAY': 'max'}).show()

df_airlines3.agg({'ARR_DELAY': 'min',
               'DEP_DELAY': 'min',
                   'CARRIER_DELAY': 'min',
                 'WEATHER_DELAY': 'min',
                 'NAS_DELAY': 'min',
                 'SECURITY_DELAY': 'min',
                 'LATE_AIRCRAFT_DELAY': 'min'}).show()


# COMMAND ----------

# Number of ORIGIN to DES pairs
df_airlines3.groupBy("ORIGIN", "DEST").count().sort(col("count").desc()).show()

# Number of delays for each pair
df_airlines3.groupBy("ORIGIN", "DEST").agg({'ARR_DELAY': 'avg',
               'DEP_DELAY': 'avg',
                'CARRIER_DELAY': 'avg',
                 'WEATHER_DELAY': 'avg',
                 'NAS_DELAY': 'avg',
                 'SECURITY_DELAY': 'avg',
                 'LATE_AIRCRAFT_DELAY': 'avg'}).show()

# COMMAND ----------

#Temporal distributions

df_airlines6.groupBy("ORIGIN","QUARTER").agg({'DEP_DELAY': 'avg', 'WEATHER_DELAY': 'avg',
                 'NAS_DELAY': 'avg',  'LATE_AIRCRAFT_DELAY': 'avg'}).orderBy(["ORIGIN", "QUARTER"], ascending=True).show()


df_airlines6.groupBy("ORIGIN","QUARTER").agg({'DEP_DELAY': 'count', 'WEATHER_DELAY': 'count',
                 'NAS_DELAY': 'count',  'LATE_AIRCRAFT_DELAY': 'count'}).orderBy(["ORIGIN", "QUARTER"], ascending=True).show()



# COMMAND ----------

display(df_airlines6.groupBy("ORIGIN","MONTH").agg(count(when(col("DEP_DEL15") == 1, True))))

# COMMAND ----------

display(df_airlines6.groupBy("ORIGIN", "DEST","MONTH").agg({'DEP_DELAY': 'avg', 'WEATHER_DELAY': 'avg',
                 'NAS_DELAY': 'avg',  'LATE_AIRCRAFT_DELAY': 'avg'}).orderBy(["ORIGIN", "DEST", "MONTH"], ascending=True))

# COMMAND ----------

display(df_airlines6.groupBy("ORIGIN","MONTH").agg({'DEP_DELAY': 'avg', 'WEATHER_DELAY': 'avg',
                 'NAS_DELAY': 'avg',  'LATE_AIRCRAFT_DELAY': 'avg'}).orderBy(["ORIGIN", "MONTH"], ascending=True))

# COMMAND ----------

#Delay Causes
#display(df_airlines6.filter( (df_airlines6.DEP_DEL15 == 1) & (df_airlines6.WEATHER_DELAY > 0))).agg(count(DEP_DEL15))

display(df_airlines6.select([col("DEP_DELAY"), (sf.when((col("DEP_DEL15") == 1) & (col("WEATHER_DELAY") > 1), True))]).agg(sf.mean("DEP_DELAY")))

display(df_airlines6.select([col("DEP_DELAY"), col("ORIGIN"), col("DEST"), (sf.when((col("DEP_DEL15") == 1) & (col("WEATHER_DELAY") > 1), True))]).groupBy("ORIGIN", "DEST").agg(sf.mean("DEP_DELAY")))


# COMMAND ----------

display(df_airlines6.select([col("DEP_DELAY"), col("ORIGIN"), col("DEST"), col("QUARTER"), (sf.when((col("DEP_DEL15") == 1) & (col("WEATHER_DELAY") > 1), True))]).groupBy("ORIGIN", "DEST", "QUARTER").agg(sf.mean("DEP_DELAY")).orderBy(["ORIGIN", "QUARTER"], ascending=True))

# COMMAND ----------

display(df_airlines6.select([col("DEP_DELAY"), col("ORIGIN"), col("DEST"), col("QUARTER"), (sf.when((col("DEP_DEL15") == 1) & (col("NAS_DELAY") > 1), True))]).groupBy("ORIGIN", "DEST", "QUARTER").agg(sf.mean("DEP_DELAY")).orderBy(["ORIGIN", "QUARTER"], ascending=True))

# COMMAND ----------

#Diverted flights
display(df_airlines6.select([sf.count(sf.when((col("DEP_DEL15") == 1) & (col("DIV_DISTANCE") > 1), True))]))
#display(df_airlines6.groupBy("ORIGIN","MONTH").agg(count(when(df_airlines6.DEP_DEL15 == 1 & df_airlines6.DIV_DISTANCE > 1, True))))

# COMMAND ----------

##I am trying to get the elapsed diverted time when there is a delay and the delay is in part due to a diversion. --- I don't think it's working
display(df_airlines6.select([col("DIV_ACTUAL_ELAPSED_TIME"), sf.when((col("DEP_DEL15") == 1) & (df_airlines6.DIV_DISTANCE.isNotNull()), True)]))

# COMMAND ----------

df_airlines_
