# Databricks notebook source
# MAGIC %md
# MAGIC # CCU059_01_D05_Hosp_Admission
# MAGIC   
# MAGIC This notebook creates tables for each of the main datasets required to create the COVID infections table. This is based off work from CCU013.
# MAGIC  
# MAGIC **Author(s)** Tom Bolton (based on original notebook by Chris Tomlinson, Johan Thygesen (inspired by Sam Hollings!))
# MAGIC
# MAGIC **Data Wrangler for CCU059(s)** Sharmin Shabnam

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

_datetimenow = datetime.datetime.now() # .strftime("%Y%m%d")
print(f"_datetimenow:  {_datetimenow}")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.sql('CLEAR CACHE')

# COMMAND ----------

# DBTITLE 1,Functions
# MAGIC %run "/Users/ss1279@leicester.ac.uk/Helper_functions/functions"

# COMMAND ----------

# MAGIC %md # 0 Parameters

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
path_cohort_2021   = f'{dbc}.{proj}_tmp_cohort_2021_mltc'
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

def print_mm_summary(df):
  a=[];b=[];c=[];d=[]
  selected = [s for s in df.columns if 'cond_' in s]
  selected.remove('cond_total')
  selected.remove('cond_concat')
  for var in selected:
    a.append(var)
    b.append(df.select(f.col(var)).filter(df[var].isNotNull()).count())
    c.append(df.agg(f.min(var)).collect()[0][0])
    d.append(df.agg(f.max(var)).collect()[0][0])

  mm_summary = pd.DataFrame({
    'mm': a,
    'count': b,
    'minimum_date': c,
    'maximum_date': d
      })

  print(mm_summary)

# COMMAND ----------

# MAGIC %md # 1 Data

# COMMAND ----------

# MAGIC %md ## 2.1 Cohort

# COMMAND ----------

spark.sql(F"""REFRESH TABLE {path_cohort_all}""")
_cohort = spark.table(path_cohort_all)#.select(['PERSON_ID', 'cond_date_COVID'])
display(_cohort)

count_var(_cohort, 'person_id')

# COMMAND ----------

# MAGIC %md ## 2.2 HES APC

# COMMAND ----------

start_date = '2020-12-01'
end_date = '2023-03-31'
path_hes_apc       = f'{dbc}.{proj}_freeze_hes_apc_{savedatetime}'
spark.sql(F"""REFRESH TABLE {path_hes_apc}""")
hes_apc  = (spark.table(path_hes_apc)
            .select(["PERSON_ID_DEID", "DIAG_4_01", "DIAG_4_CONCAT", 
                     "ADMIDATE", "DISDATE", "DISMETH", 
                     "EPISTAT", "FDE", 
                    ])
            .filter(f.col("PERSON_ID_DEID").isNotNull())
            .where((f.col('ADMIDATE') >= start_date) & (f.col('ADMIDATE') <= end_date))
           )


w_max_date = Window.partitionBy("PERSON_ID_DEID", "ADMIDATE").orderBy(f.col("DISDATE").desc_nulls_last())
_hes_apc = (hes_apc
            .filter(f.col("DISDATE").isNotNull())
            .withColumn("max_date_rank", f.row_number().over(w_max_date))
            .withColumn("max_date_flag", f.when(f.col("max_date_rank")==1, f.lit(1)).otherwise(f.lit(0)))
            .filter(f.col("max_date_flag")==1)
            .withColumn("final_discharge_date", f.lit("1800-01-01"))
            .withColumn("final_discharge_date",
                        f.when(( 
                          (f.col("max_date_flag")==1) 
                          & ((f.col("FDE")==1)
                             |((f.col("FDE")==0) & (f.col("EPISTAT")=="3") 
                               & (f.col("DISMETH").isin(["1", "2", "3", "4", "5"]))))), 
                                          f.col("DISDATE")).otherwise(f.col("final_discharge_date")))
            .withColumn("final_discharge_flag",
                        f.when(f.col("final_discharge_date")=="1800-01-01", f.lit(0)).otherwise(f.lit(1)))
            .filter(f.col("final_discharge_flag")==1)
           )

_hes_apc = (_hes_apc
            .withColumnRenamed('PERSON_ID_DEID', 'person_id') 
            .withColumnRenamed('SUSRECID', 'spell_id') 
            .join(_cohort.select(['person_id']).distinct(), ['person_id'], 'leftsemi')
            .dropDuplicates()
            .where((f.col('admidate') >= start_date) & (f.col('admidate') <= end_date))
            .where(f.col('admidate') <= f.col('disdate'))
            .orderBy(['person_id','admidate'])
            .select(['person_id','admidate','disdate', 'diag_4_01','diag_4_concat'])
           )
_hes_apc = _hes_apc.select([f.col(x).alias(x.lower()) for x in _hes_apc.columns])
save_dataset('_tmp_hes_apc', _hes_apc)
_hes_apc = read_dataset('_tmp_hes_apc')

# COMMAND ----------

# MAGIC %md ## 2.2 Death

# COMMAND ----------

# ==========================================
#              DEATHS
# ==========================================
deaths = spark.table(path_deaths)

from datetime import date, timedelta

sdate = date(1900,1,1)     # start date
edate = date(2023,3,31)   # end date
dates = pd.date_range(sdate,edate-timedelta(days=1),freq='d').strftime('%Y%m%d').tolist()
_deaths_dates_invalid = (deaths
                         .select(['REG_DATE_OF_DEATH'])
                         .distinct()
                         .filter(~f.col("REG_DATE_OF_DEATH").isin(dates))
                        )
_deaths_dates_invalid = [row.REG_DATE_OF_DEATH for row in _deaths_dates_invalid.select("REG_DATE_OF_DEATH").collect()]

start_date = '2020-01-01'
end_date = '2023-03-31'
_win = (Window
      .partitionBy('PERSON_ID')
      .orderBy(f.desc('REG_DATE'), f.desc('REG_DATE_OF_DEATH'), f.desc('S_UNDERLYING_COD_ICD10'))
           )

_deaths = (deaths
           .select(['DEC_CONF_NHS_NUMBER_CLEAN_DEID', 'REG_DATE_OF_DEATH', 'REG_DATE',
                    'S_UNDERLYING_COD_ICD10', 'S_COD_CODE_1', 
                    'S_COD_CODE_2', 'S_COD_CODE_3', 'S_COD_CODE_4', 'S_COD_CODE_5', 'S_COD_CODE_6',
                    'S_COD_CODE_7', 'S_COD_CODE_8', 'S_COD_CODE_9', 'S_COD_CODE_10', 'S_COD_CODE_11', 
                    'S_COD_CODE_12', 'S_COD_CODE_13', 'S_COD_CODE_14', 'S_COD_CODE_15'])
           .withColumnRenamed('DEC_CONF_NHS_NUMBER_CLEAN_DEID', 'PERSON_ID')
           .withColumn("REG_DATE_OF_DEATH",
                       f.when(f.col("REG_DATE_OF_DEATH").isin(_deaths_dates_invalid), f.col("REG_DATE"))
                       .otherwise(f.col("REG_DATE_OF_DEATH")))
           .withColumn('REG_DATE', f.to_date(f.col('REG_DATE'), 'yyyyMMdd'))
           .withColumn('REG_DATE_OF_DEATH', f.to_date(f.col('REG_DATE_OF_DEATH'), 'yyyyMMdd'))
           .withColumn('_rownum', f.row_number().over(_win))
           .where(
             (f.col('_rownum') == 1)
             & (f.col('PERSON_ID').isNotNull())
             & (f.col('REG_DATE_OF_DEATH').isNotNull())
             & (f.col('REG_DATE_OF_DEATH') > '1900-01-01')
             & (f.col('REG_DATE_OF_DEATH') <= '2023-03-31') 
           )
           .drop('_rownum')
           .withColumnRenamed('S_UNDERLYING_COD_ICD10', 'S_COD_CODE_0')
           .where((f.col('REG_DATE_OF_DEATH') >= start_date) & (f.col('REG_DATE_OF_DEATH') <= end_date))
           .distinct() 
          )
_deaths = (_deaths
           .withColumn('S_COD_CODE_CONCAT',
                       f.concat_ws(',', *[col for col in _deaths.columns if re.match('.*_COD_.*', col)]))
           .select(['PERSON_ID', 'REG_DATE_OF_DEATH', 'REG_DATE', 'S_COD_CODE_0', 'S_COD_CODE_CONCAT'])
           .withColumnRenamed('PERSON_ID', 'person_id') 
           .join(_cohort.select(['person_id']), ['person_id'], 'leftsemi')
          )
_deaths = _deaths.select([f.col(x).alias(x.lower()) for x in _deaths.columns])
save_dataset('_tmp_deaths', _deaths)
_deaths = read_dataset('_tmp_deaths')

# COMMAND ----------

# MAGIC %md # 3 Add Hosp Death

# COMMAND ----------

def join_hosp_death_to_cohort(_cohort, year, _hes_apc, _deaths, start_date, end_date):
  _cohort_x = _cohort.filter(f.col('year')==year)
  
  _hes_apc_x = (_hes_apc
                   .join(_cohort_x.select(['person_id']), ['person_id'], 'leftsemi')
                   .where((f.col('admidate') >= start_date) & (f.col('admidate') <= end_date))
                   .orderBy(['person_id','admidate'])
                   .select(['person_id','admidate'])
                   .distinct()
                   .groupBy('person_id').count().select('person_id', f.col('count').alias('num_hosp_adm'))
             )

  _deaths_x = (_deaths
                   .join(_cohort_x.select(['person_id']), ['person_id'], 'leftsemi')
                   .where((f.col('reg_date_of_death') >= start_date) & (f.col('reg_date_of_death') <= end_date))
                   .orderBy(['person_id','reg_date_of_death'])
                   .select(['person_id','reg_date_of_death'])
                   .distinct()
                  .withColumn('deaths', f.lit(1))
             )
 
  _cohort_x = (_cohort_x
                .join(_hes_apc_x, on='person_id', how='left')
                .join(_deaths_x, on='person_id', how='left')
                .fillna(value=0, subset=['num_hosp_adm', 'deaths'])
               .withColumn('followup_start_date', f.col('start_date'))
               .withColumn('followup_end_date',
                    f.when(((f.col('dod').isNotNull()) & (f.col('dod') <= f.col('end_date'))), f.col('dod'))
                      .otherwise(f.col('end_date'))
                     )
               .drop('censor_start_date','censor_end_date')
               .withColumn("followup_yrs", 
                           f.datediff(f.col("followup_end_date"),f.col("followup_start_date"))/365.2)
               .withColumn("followup_yrs", f.round(f.col('followup_yrs'), 2))
               )
  splits = [('<=45', 45), ('46-60', 60), ('61-75', 75), ('>75', float('Inf'))]
  bins = reduce(lambda c, i: c.when(f.col('age_at_start') <= i[1], i[0]), splits,
                f.when(f.col('age_at_start') < splits[0][0], None)).otherwise(splits[-1][0]).alias('age_cat')
  _cohort_x = _cohort_x.select(*_cohort_x.columns, bins)
  _cohort_x = (_cohort_x
               .withColumnRenamed('imd_2019_quintiles', 'imd')
               .withColumnRenamed('sex', 'sex_female')
               .filter((f.col("sex") != 9) & (f.col("sex") != 0))
               .withColumn("sex_female", f.when(f.col('sex_female') == 2, 1).otherwise(f.lit(0)))
               .filter(f.col("imd").isNotNull())
               .filter((f.col("ethnic_cat")!='Unknown'))
              )
  count_var(_cohort_x, 'person_id')
  print('num_hosp_adm', _cohort_x.select(['num_hosp_adm']).groupBy().sum().collect()[0][0]) 
  print('deaths', _cohort_x.select(['deaths']).groupBy().sum().collect()[0][0]) 
  return _cohort_x


# COMMAND ----------

# MAGIC %md ## 3.1 2021

# COMMAND ----------

start_date = '2021-12-01'
end_date = '2022-03-31'
_cohort_2021 = join_hosp_death_to_cohort(_cohort, 2021, _hes_apc, _deaths, start_date, end_date)
save_dataset('_tmp_cohort_2021_mltc_hosp_death', _cohort_2021)
_cohort_2021 = read_dataset('_tmp_cohort_2021_mltc_hosp_death')

