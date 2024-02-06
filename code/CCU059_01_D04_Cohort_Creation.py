# Databricks notebook source
# MAGIC %md # CCU059_01_D04_Cohort_Creation
# MAGIC  
# MAGIC **Description** This notebook runs through the inclusion/exclusion criteria for the skinny cohort after QA.
# MAGIC
# MAGIC **Original Author(s)** Jenny Cooper, Samantha Ip
# MAGIC
# MAGIC **Data Wrangler for CCU059(s)** Sharmin Shabnam

# COMMAND ----------

# MAGIC %md # 1 Import Functions

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window
from pyspark.sql.functions import isnan, when, count, col
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
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.sql('CLEAR CACHE')
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 10)
pd.set_option('expand_frame_repr', False)

# COMMAND ----------

# MAGIC %run "/Workspaces/dars_nic_391419_j3w9t_collab/CCU059/Helper_functions/functions"

# COMMAND ----------

# MAGIC %md # 2 Data

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
savedatetime = "20230126"
# in tables (available post table_freeze)
path_deaths        = f'{dbc}.{proj}_freeze_deaths_{savedatetime}'
path_gdppr         = f'{dbc}.{proj}_freeze_gdppr_{savedatetime}'
path_hes_apc       = f'{dbc}.{proj}_freeze_hes_apc_{savedatetime}'
path_codelist      = f'{dbc}.{proj}_codelists_exp_wo_dupes_20230428' 
path_skinny        = f'{dbc}.{proj}_final_skinny_assembled'
save_data = True

# COMMAND ----------

def save_dataset(name, dataset):
  proj = 'ccu059_01'
  db = 'dars_nic_391419_j3w9t'
  dbc = f'{db}_collab'
  
  outName = f'{proj}{name}'.lower()
  dataset.createOrReplaceGlobalTempView(outName)
  drop_table(outName, if_exists=True)
  create_table(outName, select_sql_script=f"SELECT * FROM global_temp.{outName}")
  
def read_dataset(name):
  proj = 'ccu059_01'
  db = 'dars_nic_391419_j3w9t'
  dbc = f'{db}_collab'
  spark.sql('CLEAR CACHE')
  path_tmp_covid = f'{dbc}.{proj}{name}'
  spark.sql(F"""REFRESH TABLE {path_tmp_covid}""")
  dataset = spark.table(path_tmp_covid)
  return dataset

# COMMAND ----------

spark.sql(F"""REFRESH TABLE {path_skinny}""")
skinny_data        = spark.table(path_skinny)
display(skinny_data)

# COMMAND ----------

count_var(skinny_data, 'PERSON_ID')
skinny_data.agg(f.min("DOB"), f.max("DOB"), f.min("DOD"), f.max("DOD")).show()
null_counts = {col:skinny_data.filter(skinny_data[col].isNull()).count() for col in skinny_data.columns}
null_counts

# COMMAND ----------

# MAGIC %md # 3 Cohort build

# COMMAND ----------

#Date parameters
start_date = '2021-12-01'
end_date = '2022-03-31'


# ==========================================
#              Cohort 2021-2022 
# ==========================================
_cohort_2021 = (skinny_data
              .filter(f.col("DOB").isNotNull())
              .filter((f.col("DOD").isNull()) | (f.col("DOD") > start_date))
              .withColumn('AGE_AT_START', f.datediff(f.to_date(f.lit(start_date), f.col('DOB'))/365.25)
              .withColumn("AGE_AT_START", f.round(f.col("AGE_AT_START"), 0))
              .filter((f.col("AGE_AT_START") >= 18) & (f.col("AGE_AT_START") <= 120))
              .orderBy('DOB')
               .withColumn("START_DATE", f.to_date(f.lit(start_date)))
               .withColumn("END_DATE", f.to_date(f.lit(end_date))
               .withColumn('CENSOR_START_DATE', f.date_add(f.col('DOB'), -0))
               .withColumn('CENSOR_END_DATE',
                      f.when(
                        (f.col('DOD').isNotNull()),
                        f.col('DOD'))
                      .otherwise(f.to_date(f.lit(end_date)))
                     )
               .drop('LSOA', 'ETHNIC', 'ETHNIC_DESC', 'region', 'IMD_2019_DECILES')
              )
_cohort_2021 = _cohort_2021.select([f.col(x).alias(x.lower()) for x in _cohort_2021.columns])
count_var(_cohort_2021, 'person_id')

# COMMAND ----------

_cohort_2021 = _cohort_2021.filter(f.col('in_gdppr')==1).drop('in_gdppr')
count_var(_cohort_2021, 'person_id')

# COMMAND ----------

def custom_round(x, base=5):
    return int(base * round(float(x)/base))

def tab_variable(df, col):
  tmp = (df
   .select(f.col(col)).groupBy(col)
   .count()
   .withColumn('total', f.sum('count').over(Window.partitionBy()))
   .withColumn('percent', f.col('count')*100/f.col('total'))
   .withColumn('percent', f.round(f.col('percent'), 4))
   .withColumn("percent", f.concat(f.lit("("), f.col('percent'), f.lit('%)')))
         .withColumn("context", f.lit(''))
  )
  
  schema = tmp.schema
  pandas_df = tmp.toPandas()
  pandas_df.loc[0, 'context'] = 'The first column is the variables categories, rest of the columns are counts of patients in each category, total patients and their percentage'

  pandas_df['count'] = pandas_df['count'].apply(lambda x: custom_round(x, base=5)).astype(int)
  pandas_df['total'] = pandas_df['total'].apply(lambda x: custom_round(x, base=5)).astype(int)
  
  if col =='imd_2019_quintiles':
    schema = t.StructType([
              t.StructField('imd_2019_quintiles', t.FloatType(), True),
              t.StructField('count', t.LongType(),True),
              t.StructField('total', t.LongType(),True),
              t.StructField('percent', t.StringType(),True),
              t.StructField('context', t.StringType(),True)
            ])
  tmp = spark.createDataFrame(pandas_df,schema=schema)
  del pandas_df
  display(tmp)
  return tmp

# COMMAND ----------

tab_variable(_cohort_2021, 'sex')

# COMMAND ----------

tab_variable(_cohort_2021, 'ethnic_cat')

# COMMAND ----------

tab_variable(_cohort_2021, 'imd_2019_quintiles')

# COMMAND ----------

_cohort_2021 = (_cohort_2021
                .withColumnRenamed('imd_2019_quintiles', 'imd')
                .withColumnRenamed('sex', 'sex_female')
                .filter((f.col("sex") != 9) & (f.col("sex") != 0))
                .withColumn("sex_female", f.when(f.col('sex_female') == 2, 1).otherwise(f.lit(0)))
                .filter(f.col("imd").isNotNull())
                .filter((f.col("ethnic_cat")!='Unknown'))
               )
count_var(_cohort_2021, 'person_id')

# COMMAND ----------

# MAGIC %md ## 3.0 Save

# COMMAND ----------

save_dataset('_tmp_cohort_2021', _cohort_2021)
_cohort_2021 = read_dataset('_tmp_cohort_2021')
