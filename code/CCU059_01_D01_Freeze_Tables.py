# Databricks notebook source
# MAGIC %md # 1 Import Functions

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW DATABASES

# COMMAND ----------

spark.sql('CLEAR CACHE')

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

from functools import reduce

import databricks.koalas as ks
import pandas as pd
import numpy as np

import re
import io
import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns

_datetimenow = datetime.datetime.now()
print(f"_datetimenow:  {_datetimenow}")

# COMMAND ----------

# MAGIC %run "/Users/ss1279@leicester.ac.uk/Helper_functions/functions"

# COMMAND ----------

# MAGIC %md # 2 Test

# COMMAND ----------

db_collab = "dars_nic_391419_j3w9t_collab"
gdppr_arch = "gdppr_dars_nic_391419_j3w9t_archive" 
hes_apc_arch = "hes_apc_all_years_archive"
chess_arch = "chess_dars_nic_391419_j3w9t_archive"
death_arch = "deaths_dars_nic_391419_j3w9t_archive"


# Load archive datasets, PySpark method
gdppr_arch = spark.sql(f'select * from {db_collab}.{gdppr_arch}')
hes_apc_arch = spark.sql(f'select * from {db_collab}.{hes_apc_arch}')
chess_arch = spark.sql(f'select * from {db_collab}.{chess_arch}')
death_arch = spark.sql(f'select * from {db_collab}.{death_arch}')

# COMMAND ----------

display(death_arch)

# COMMAND ----------

#This funciton is in the notebook date_management called with %run above
def print_production_date(df, name):
  production_dates = list_of_production_dates(df, "ProductionDate")
  print(f"Latest production date for {name} : {max(production_dates)}")

print_production_date(gdppr_arch, 'gdppr_arch')
print_production_date(hes_apc_arch, 'hes_apc_arch')
print_production_date(chess_arch, 'chess_arch')
print_production_date(death_arch, 'death_arch')

# Latest production date for gdppr_arch : 2023-07-31 16:50:42.226641
# Latest production date for hes_apc_arch : 2023-06-21 10:30:25.748339
# Latest production date for chess_arch : 2023-03-31 00:00:00.000000
# Latest production date for death_arch : 2023-07-31 16:50:42.226641

# COMMAND ----------

# MAGIC %md # 3 Table Parameters

# COMMAND ----------

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
proj = 'ccu059_01'


# -----------------------------------------------------------------------------
# Databases
# -----------------------------------------------------------------------------
db = 'dars_nic_391419_j3w9t'
dbc = f'{db}_collab'


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# archive tables
tmp_archive_date = '2022-12-31'
data = [
    ['deaths',      dbc, f'deaths_{db}_archive',        tmp_archive_date, 'DEC_CONF_NHS_NUMBER_CLEAN_DEID', 'REG_DATE_OF_DEATH']
  , ['gdppr',       dbc, f'gdppr_{db}_archive',         tmp_archive_date, 'NHS_NUMBER_DEID', 'DATE']
  , ['hes_apc',     dbc, f'hes_apc_all_years_archive',  tmp_archive_date, 'PERSON_ID_DEID',  'EPISTART']   
]
df_archive = pd.DataFrame(data, columns = ['dataset', 'database', 'table', 'production_date', 'id_var', 'date_var'])
df_archive

# COMMAND ----------

# MAGIC %md # 4 Freeze Tables

# COMMAND ----------


spark.conf.set('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true')
datetimenow = "20230126" #datetime.datetime.now().strftime("%Y%m%d")

for ind in df_archive.index:
  row = df_archive.iloc[ind]
  
  tabpath = row['database'] + '.' + row['table'] 
  prodDate = row['production_date']
  id = row['id_var']  

  dataset_name = row['dataset']
  table_name = f'{proj}_freeze_{dataset_name}_{datetimenow}'  
  
  #get archive table
  tmp = spark.table(tabpath).where(f.col('ProductionDate').startswith(prodDate))  
  tmp.createOrReplaceGlobalTempView(f"{proj}_{dataset_name}")
  drop_table(table_name, if_exists=True)
  create_table(table_name , select_sql_script=f"SELECT * FROM global_temp.{proj}_{dataset_name}") 

  person_id_deid = row['id_var']
  print(f'{0 if ind<10 else ""}' + str(ind) + ' ' + tabpath + ' (' + prodDate + ')' + ' [' + table_name + ']', flush=True)
  count_var(tmp, person_id_deid)

# COMMAND ----------

for v in list(vars().keys()):
  if(re.search("dataframe", type(vars()[v]).__name__.lower())):
    print(v, type(vars()[v]))


