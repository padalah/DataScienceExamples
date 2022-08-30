# Databricks notebook source
#Read airport data (Lea)

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
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

#Aiprort and their timezones
df_airlinesAll = spark.read.parquet(f"{blob_url}/lea/parquet_continental_data")
display(df_airlinesAll)

# COMMAND ----------

#Filter out columns that are not helpful for flight based algorithm analysis
df_airlinesAll.columns

# COMMAND ----------

df_airlinesAll = df_airlinesAll.drop('DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN_STATE_FIPS', 'ORIGIN_WAC', 'DEST_STATE_FIPS', 'DEST_WAC')
