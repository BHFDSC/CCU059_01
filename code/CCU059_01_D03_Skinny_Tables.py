# Databricks notebook source
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
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.sql('CLEAR CACHE')

# COMMAND ----------

# MAGIC %md # 1 Import Functions

# COMMAND ----------

# MAGIC %run "/Users/ss1279@leicester.ac.uk/Helper_functions/skinny"

# COMMAND ----------

# MAGIC %run "/Users/ss1279@leicester.ac.uk/Helper_functions/functions"

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
path_hes_ae        = f'{dbc}.{proj}_freeze_hes_ae_{savedatetime}'
path_hes_op        = f'{dbc}.{proj}_freeze_hes_op_{savedatetime}'
path_codelist      = f'{dbc}.ccu010_01_codelists'
save_data = False

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

# MAGIC %md # 3 Tables

# COMMAND ----------

hes_apc         = spark.table(path_hes_apc)
hes_op          = spark.table(path_hes_op)
hes_ae          = spark.table(path_hes_ae)
gdppr           = spark.table(path_gdppr)
deaths          = spark.table(path_deaths)

# COMMAND ----------

# MAGIC %md # 4 Skinny Unassembled

# COMMAND ----------

# MAGIC %md ## 4.1 Define Functions

# COMMAND ----------

from datetime import date, timedelta

sdate = date(1900,1,1)     # start date
edate = date(2022,10,31)   # end date
dates = pd.date_range(sdate,edate-timedelta(days=1),freq='d').strftime('%Y%m%d').tolist()
_deaths_dates_invalid = (deaths
                         .select(['REG_DATE_OF_DEATH'])
                         .distinct()
                         .filter(~f.col("REG_DATE_OF_DEATH").isin(dates))
                        )
_deaths_dates_invalid = [row.REG_DATE_OF_DEATH for row in _deaths_dates_invalid.select("REG_DATE_OF_DEATH").collect()]

# COMMAND ----------

# note - could harmonise LSOA01 to LSOA11

def skinny_unassembled(_hes_apc, _hes_ae, _hes_op, _gdppr, _deaths, _deaths_dates_invalid, _print_id=False):
  
  # ------------------------------------------------------------------------------------
  # _hes_apc
  # ------------------------------------------------------------------------------------
  _hes_apc = _hes_apc\
    .withColumn('_id', f.monotonically_increasing_id())\
    .select('PERSON_ID_DEID', 'EPIKEY', 'EPISTART', 'MYDOB', 'SEX', 'ETHNOS', 'LSOA11', '_id')\
    .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')\
    .withColumnRenamed('EPIKEY', 'RECORD_ID')\
    .withColumnRenamed('EPISTART', 'RECORD_DATE')\
    .withColumnRenamed('MYDOB', 'DOB')\
    .withColumnRenamed('ETHNOS', 'ETHNIC')\
    .withColumnRenamed('LSOA11', 'LSOA')\
    .withColumn('DOB', f.to_date('DOB', 'MMyyyy'))\
    .withColumn('DOD', f.lit(None))\
    .withColumn('RECORD_PRIMARY', f.lit(0))\
    .withColumn('RECORD_SOURCE', f.lit('hes_apc'))\
    .select('PERSON_ID', 'RECORD_SOURCE', 'RECORD_PRIMARY', 'RECORD_ID', 'RECORD_DATE', 'DOB', 'SEX', 'ETHNIC', 'LSOA', 'DOD', '_id')\
    .distinct()
  
  if (_print_id):
    count_var(_hes_apc, 'PERSON_ID')
  
  # ------------------------------------------------------------------------------------
  # _hes_ae  
  # ------------------------------------------------------------------------------------
  # would like to change 365 to 365.25
  _hes_ae = _hes_ae\
    .withColumn('_id', f.monotonically_increasing_id())\
    .select('PERSON_ID_DEID', 'EPIKEY', 'AEKEY', 'ARRIVALDATE', 'ARRIVALAGE_CALC', 'SEX', 'ETHNOS', 'LSOA11', '_id')\
    .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')\
    .withColumnRenamed('ARRIVALDATE', 'RECORD_DATE')\
    .withColumnRenamed('ETHNOS', 'ETHNIC')\
    .withColumnRenamed('LSOA11', 'LSOA')\
    .withColumn('RECORD_ID', f.coalesce('EPIKEY', 'AEKEY'))\
    .withColumn('DOB', f.to_date(f.date_trunc('MM', f.expr('date_add(RECORD_DATE, -ARRIVALAGE_CALC*365)')), 'yyyy-MM-dd'))\
    .drop('EPIKEY', 'AEKEY', 'ARRIVALAGE_CALC')\
    .withColumn('DOD', f.lit(None))\
    .withColumn('RECORD_PRIMARY', f.lit(0))\
    .withColumn('RECORD_SOURCE', f.lit('hes_ae'))\
    .select('PERSON_ID', 'RECORD_SOURCE', 'RECORD_PRIMARY', 'RECORD_ID', 'RECORD_DATE', 'DOB', 'SEX', 'ETHNIC', 'LSOA', 'DOD', '_id')\
    .distinct()

  if (_print_id):
    count_var(_hes_ae, 'PERSON_ID')
  
  
  # ------------------------------------------------------------------------------------
  # _hes_op
  # ------------------------------------------------------------------------------------
  _hes_op = _hes_op\
    .withColumn('_id', f.monotonically_increasing_id())\
    .select('PERSON_ID_DEID', 'ATTENDKEY', 'APPTDATE', 'APPTAGE_CALC', 'SEX', 'ETHNOS', 'LSOA11', '_id')\
    .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')\
    .withColumnRenamed('ATTENDKEY', 'RECORD_ID')\
    .withColumnRenamed('APPTDATE', 'RECORD_DATE')\
    .withColumnRenamed('ETHNOS', 'ETHNIC')\
    .withColumnRenamed('LSOA11', 'LSOA')\
    .withColumn('DOB', f.to_date(f.date_trunc('MM', f.expr('date_add(RECORD_DATE, -APPTAGE_CALC*365)')), 'yyyy-MM-dd'))\
    .drop('APPTAGE_CALC')\
    .withColumn('DOD', f.lit(None))\
    .withColumn('RECORD_PRIMARY', f.lit(0))\
    .withColumn('RECORD_SOURCE', f.lit('hes_op'))\
    .select('PERSON_ID', 'RECORD_SOURCE', 'RECORD_PRIMARY', 'RECORD_ID', 'RECORD_DATE', 'DOB', 'SEX', 'ETHNIC', 'LSOA', 'DOD', '_id')\
    .distinct()

  if (_print_id):
    count_var(_hes_op, 'PERSON_ID')

   
  
  # ------------------------------------------------------------------------------------
  # _gdppr  
  # ------------------------------------------------------------------------------------
  _gdppr = _gdppr\
    .withColumn('_id', f.monotonically_increasing_id())
  _gdppr_orig = _gdppr    
  _gdppr = _gdppr\
    .select('NHS_NUMBER_DEID', 'REPORTING_PERIOD_END_DATE', 'YEAR_OF_BIRTH', 'SEX', 'ETHNIC', 'LSOA', 'YEAR_OF_DEATH', '_id')\
    .withColumnRenamed('NHS_NUMBER_DEID', 'PERSON_ID')\
    .withColumnRenamed('REPORTING_PERIOD_END_DATE', 'RECORD_DATE')\
    .withColumnRenamed('YEAR_OF_BIRTH', 'DOB')\
    .withColumn('DOB', f.to_date(f.col('DOB').cast(t.StringType()), "yyyy"))\
    .withColumnRenamed('YEAR_OF_DEATH', 'DOD')\
    .withColumn('DOD', f.to_date(f.col('DOD').cast(t.StringType()), "yyyy"))\
    .withColumn('RECORD_ID', f.lit(None))\
    .withColumn('RECORD_PRIMARY', f.lit(1))\
    .withColumn('RECORD_SOURCE', f.lit('gdppr'))\
    .select('PERSON_ID', 'RECORD_SOURCE', 'RECORD_PRIMARY', 'RECORD_ID', 'RECORD_DATE', 'DOB', 'SEX', 'ETHNIC', 'LSOA', 'DOD', '_id')\
    .distinct()
  
  if (_print_id):
    count_var(_gdppr, 'PERSON_ID')

  
  # ------------------------------------------------------------------------------------
  # _gdppr_snomed_ethnic
  # ------------------------------------------------------------------------------------
  _dss_mapping_snomed_ethnic = spark.table('dss_corporate.gdppr_ethnicity_mappings')\
    .select('ConceptId', 'PrimaryCode')\
    .withColumnRenamed('ConceptId', 'CODE')\
    .withColumnRenamed('PrimaryCode', 'ETHNIC')
  assert _dss_mapping_snomed_ethnic.select('CODE').distinct().count() == _dss_mapping_snomed_ethnic.select('CODE').count() 
  
  # notes:
  # - remove SEX, DOB, DOD? appropriate to add here?
  #     this information is added above with RECORD_DATE=REPORTING_PERIOD_END_DATE
  #     should we only be adding ethnic here with RECORD_DATE=DATE?
  _gdppr_snomed_ethnic = _gdppr_orig\
    .select('NHS_NUMBER_DEID', 'DATE', 'CODE', 'YEAR_OF_BIRTH', 'SEX', 'YEAR_OF_DEATH', '_id')\
    .join(_dss_mapping_snomed_ethnic, on='CODE', how='inner')\
    .drop('CODE')\
    .withColumnRenamed('NHS_NUMBER_DEID', 'PERSON_ID')\
    .withColumnRenamed('DATE', 'RECORD_DATE')\
    .withColumnRenamed('YEAR_OF_BIRTH', 'DOB')\
    .withColumn('DOB', f.to_date(f.col('DOB').cast(t.StringType()), "yyyy"))\
    .withColumnRenamed('YEAR_OF_DEATH', 'DOD')\
    .withColumn('DOD', f.to_date(f.col('DOD').cast(t.StringType()), "yyyy"))\
    .withColumn('LSOA', f.lit(None))\
    .withColumn('RECORD_ID', f.lit(None))\
    .withColumn('RECORD_PRIMARY', f.lit(1))\
    .withColumn('RECORD_SOURCE', f.lit('gdppr_snomed'))\
    .select('PERSON_ID', 'RECORD_SOURCE', 'RECORD_PRIMARY', 'RECORD_ID', 'RECORD_DATE', 'DOB', 'SEX', 'ETHNIC', 'LSOA', 'DOD', '_id')\
    .distinct()
  
  if (_print_id):
    count_var(_gdppr_snomed_ethnic, 'PERSON_ID')

  
  # ------------------------------------------------------------------------------------
  # _deaths
  # ------------------------------------------------------------------------------------
  # - reformat dates
  # - reduce to single death per individual
  from datetime import date, timedelta

  sdate = date(1900,1,1)     # start date
  edate = date(2022,10,31)   # end date
  dates = pd.date_range(sdate,edate-timedelta(days=1),freq='d').strftime('%Y%m%d').tolist()
  _deaths_dates_invalid = (deaths
                           .select(['REG_DATE_OF_DEATH'])
                           .distinct()
                           .filter(~f.col("REG_DATE_OF_DEATH").isin(dates))
                          )
  _deaths_dates_invalid = [row.REG_DATE_OF_DEATH for row in _deaths_dates_invalid.select("REG_DATE_OF_DEATH").collect()]
 
  _win = (Window
      .partitionBy('PERSON_ID')
      .orderBy(f.desc('REG_DATE'), f.desc('REG_DATE_OF_DEATH'), f.desc('S_UNDERLYING_COD_ICD10'))
           )
  _deaths = (deaths
      .withColumn('_id', f.monotonically_increasing_id())
      .select('DEC_CONF_NHS_NUMBER_CLEAN_DEID', 'REG_DATE', 'REG_DATE_OF_DEATH', 'S_UNDERLYING_COD_ICD10', '_id')
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
        & (f.col('REG_DATE_OF_DEATH') <= '2022-10-31') 
      )
      .withColumn('DOB', f.lit(None))
      .withColumn('SEX', f.lit(None))
      .withColumn('ETHNIC', f.lit(None))
      .withColumn('LSOA', f.lit(None))
      .withColumnRenamed('REG_DATE_OF_DEATH', 'DOD')
      .withColumnRenamed('REG_DATE', 'RECORD_DATE')
      .withColumn('RECORD_ID', f.lit(None))
      .withColumn('RECORD_PRIMARY', f.lit(0))
      .withColumn('RECORD_SOURCE', f.lit('deaths'))
      .select('PERSON_ID', 'RECORD_SOURCE', 'RECORD_PRIMARY', 'RECORD_ID', 'RECORD_DATE',
              'DOB', 'SEX', 'ETHNIC', 'LSOA', 'DOD', '_id')
      .distinct() 
            )
  
  if (_print_id):
    count_var(_deaths, 'PERSON_ID')

  # ------------------------------------------------------------------------------------    
  # _unassembled
  # ------------------------------------------------------------------------------------
  # union all
  _unassembled = _hes_apc\
    .unionByName(_hes_ae)\
    .unionByName(_hes_op)\
    .unionByName(_gdppr)\
    .unionByName(_gdppr_snomed_ethnic)\
    .unionByName(_deaths)
  
  if (_print_id):
    count_var(_unassembled, 'PERSON_ID')

  return _unassembled

# COMMAND ----------

# MAGIC %md ## 4.2 Save Data

# COMMAND ----------

# 5 hours
# Note: individual censor dates option used
# prioritise primary care records turned off - happy to take the most recent HES (delivery) record

unassembled = skinny_unassembled(_hes_apc=hes_apc, _hes_ae=hes_ae, _hes_op=hes_op,
                                 _gdppr=gdppr, _deaths=deaths, _deaths_dates_invalid =_deaths_dates_invalid,
                                 _print_id=False)

if save_data:
  save_dataset('_tmp_skinny_unassembled', unassembled)
unassembled = read_dataset('_tmp_skinny_unassembled')

count_var(unassembled, 'PERSON_ID')

# COMMAND ----------

display(unassembled)

# COMMAND ----------

# MAGIC %md # 5 Skinny Assembled

# COMMAND ----------

# MAGIC %md ## 5.1 Define function

# COMMAND ----------

def skinny_assembled(_unassembled, _overall_censor_date=None, _individual_censor_dates=None, _prioritise_primary_care=1):
  
  # check
  assert _prioritise_primary_care in [0, 1]
  
  # remove those with missing ID (performed at this stage to ensure unassembled output agrees with that of the wranglers)
  _unassembled = _unassembled.where(f.col('PERSON_ID').isNotNull())
  
  # make a copy before filtering that can be called for DOD and in_gdppr below
  _unassembled_orig = _unassembled  
    
  # ------------------------------------------------------------------------------------
  # filter
  # ------------------------------------------------------------------------------------   
  # NOTE: not removing dataset = "gdppr_snomed", since this was inadvertently included in previous code
  # check only one of overall censor date and individual censor dates has been specified
  
  if(_overall_censor_date):
    print(f'** applying overall censor date = {_overall_censor_date} **')
    
    _unassembled_orig = _unassembled  
    
    _unassembled = _unassembled\
      .where(f.col('RECORD_DATE') < f.to_date(f.lit(_overall_censor_date)))

  elif(_individual_censor_dates):
    # ensure dataframe format
    #   TODO - check individual censor dates is correct format
    #     e.g., column named PERSON_ID, single row per person, column named CENSOR_DATE, non-missing
    #     counts
    #     date format, yyyy-MM-dd
    
    assert set(_individual_censor_dates.columns).issubset(set(['PERSON_ID', 'CENSOR_DATE'])) 
    assert _individual_censor_dates.count() == _individual_censor_dates.select('PERSON_ID').distinct().count()
    assert _individual_censor_dates.where(f.col('PERSON_ID').isNull()).count() == 0
    assert _individual_censor_dates.where(f.col('CENSOR_DATE').isNull()).count() == 0
    assert dict(_individual_censor_dates.dtypes)['CENSOR_DATE'] == 'date'
    
    print(f'** applying individual censor dates **')
    _unassembled = _unassembled\
      .join(_individual_censor_dates, on=['PERSON_ID'], how='inner')
    
    _unassembled_orig = _unassembled  
    
    _unassembled = _unassembled\
      .where(f.col('RECORD_DATE') <= f.col('CENSOR_DATE'))
  
  else:
    print(f'** no censor date(s) applied **')
  
  #count_var(_unassembled, 'PERSON_ID')

  # ------------------------------------------------------------------------------------
  # _prioritise_primary_care
  # ------------------------------------------------------------------------------------       
  if(_prioritise_primary_care == 0):
    # if zero (no) then null the RECORD_PRIMARY column
    # note: this avoids having to redefine windows
    print(f'** NOT prioritising primary care records **')
    _unassembled = _unassembled.withColumn('RECORD_PRIMARY', f.lit(None))
  
  
  # ------------------------------------------------------------------------------------
  # characteristic selection preparation
  # ------------------------------------------------------------------------------------ 
  # define RECORD_SOURCEn
  _unassembled = _unassembled\
    .withColumn('RECORD_SOURCEn',\
      f.when(f.col('RECORD_SOURCE') == 'hes_apc', 1)\
      .when(f.col('RECORD_SOURCE') == 'hes_op', 2)\
      .when(f.col('RECORD_SOURCE') == 'hes_ae', 3)\
      .when(f.col('RECORD_SOURCE') == 'gdppr', 4)\
      .when(f.col('RECORD_SOURCE') == 'gdppr_snomed', 5)\
      .when(f.col('RECORD_SOURCE') == 'deaths', 6)\
    )
  # assert _unassembled.count() == _unassembled.where(f.col('RECORD_SOURCEn').isNotNull()).count()
  
  
  # define windows for _rownum
  _win_rownum_DOB = Window\
    .partitionBy('PERSON_ID')\
    .orderBy(['DOB_null', f.desc('RECORD_PRIMARY'), f.desc('RECORD_DATE'), 'RECORD_SOURCEn', 'RECORD_ID', '_id'])
  _win_rownum_SEX = Window\
    .partitionBy('PERSON_ID')\
    .orderBy(['SEX_null', f.desc('RECORD_PRIMARY'), f.desc('RECORD_DATE'), 'RECORD_SOURCEn', 'RECORD_ID', '_id'])
  _win_rownum_ETHNIC = Window\
    .partitionBy('PERSON_ID')\
    .orderBy(['ETHNIC_null', f.desc('RECORD_PRIMARY'), f.desc('RECORD_DATE'), 'RECORD_SOURCEn', 'RECORD_ID', '_id'])
  # note: not prioritising primary records for LSOA
  _win_rownum_LSOA = Window\
    .partitionBy('PERSON_ID')\
    .orderBy(['LSOA_null', f.desc('RECORD_DATE'), 'RECORD_SOURCEn', 'RECORD_ID', '_id']) 
  _win_rownum_DOD = Window\
    .partitionBy('PERSON_ID')\
    .orderBy([f.desc('RECORD_DEATHS'), f.desc('RECORD_DATE'), '_id'])

  
  # create null indicators and row numbers based on ordering above
  #   add further null info for LSOA? 
  _unassembled = _unassembled\
    .withColumn('DOB_null', f.when(\
      (f.col('DOB').isNull())\
      | (f.trim(f.col('DOB')).isin(['']))\
      | (f.col('DOB') < '1900-01-01')\
      | (f.col('DOB') > f.current_date())\
      | (f.col('DOB') > f.col('RECORD_DATE'))\
      , 1).otherwise(0)\
    )\
    .withColumn('SEX_null', f.when(\
      (f.col('SEX').isNull())\
      | (f.trim(f.col('SEX')).isin(['', '9', '0']))\
      , 1).otherwise(0)\
    )\
    .withColumn('ETHNIC_null', f.when(\
      (f.col('ETHNIC').isNull())\
      | (f.trim(f.col('ETHNIC')).isin(['', '9', '99', 'X', 'Z']))\
      , 1).otherwise(0)\
    )\
    .withColumn('LSOA_null', f.when(\
      (f.col('LSOA').isNull())\
      | (f.trim(f.col('LSOA')).isin(['']))\
      , 1).otherwise(0)\
    )\
    .withColumn('DOD_null', f.when(\
      (f.col('DOD').isNull())\
      | (f.trim(f.col('DOD')).isin(['']))\
      | (f.col('DOD') < '1900-01-01')\
      | (f.col('DOD') > f.current_date())\
      | (f.col('DOD') > f.col('RECORD_DATE'))\
      , 1).otherwise(0)\
    )\
    .withColumn('_rownum_DOB', f.row_number().over(_win_rownum_DOB))\
    .withColumn('_rownum_SEX', f.row_number().over(_win_rownum_SEX))\
    .withColumn('_rownum_ETHNIC', f.row_number().over(_win_rownum_ETHNIC))\
    .withColumn('_rownum_LSOA', f.row_number().over(_win_rownum_LSOA))\

  
  # ------------------------------------------------------------------------------------
  # ties
  # ------------------------------------------------------------------------------------   
  # create indicators for tied records where variable differs
  _vlist = ['DOB', 'SEX', 'ETHNIC', 'LSOA']
  for ind, var in enumerate(_vlist):
    # define window for tied records 
    _win_tie = Window\
      .partitionBy('PERSON_ID')\
      .orderBy(f'{var}_null', f.desc('RECORD_PRIMARY'), f.desc('RECORD_DATE'))
    if var == 'LSOA':
      _win_tie = Window\
        .partitionBy('PERSON_ID')\
        .orderBy(f'{var}_null', f.desc('RECORD_DATE'))
      
    # count distinct values of var (including null) within tied records
    _tmp_tie = _unassembled\
      .withColumn(f'_tie_{var}', f.dense_rank().over(_win_tie))\
      .where(f.col(f'_tie_{var}') == 1)\
      .groupBy('PERSON_ID')\
      .agg(\
        f.countDistinct(f.col(f'{var}')).alias(f'_n_distinct_{var}')\
        , f.countDistinct(f.when(f.col(f'{var}').isNull(), 1)).alias(f'_null_{var}')\
      )\
      .withColumn(f'_tie_{var}', f.when((f.col(f'_n_distinct_{var}') + f.col(f'_null_{var}')) > 1, 1).otherwise(0))\
      .select('PERSON_ID', f'_tie_{var}')
  
    # join var
    if(ind == 0): _ties = _tmp_tie
    else: 
      _ties = _ties\
        .join(_tmp_tie, on=['PERSON_ID'], how='outer')    
  
  # ------------------------------------------------------------------------------------ 
  # DOD
  # ------------------------------------------------------------------------------------ 
  _win_tie_DOD = Window\
    .partitionBy('PERSON_ID')\
    .orderBy(f.desc('RECORD_DEATHS'), f.desc('RECORD_DATE'))
  
  _tmp_tie_DOD = _unassembled_orig\
    .withColumn('RECORD_DEATHS', f.when(f.col('RECORD_SOURCE') == 'deaths', 1).otherwise(0))\
    .withColumn('_tie_DOD', f.dense_rank().over(_win_tie_DOD))\
    .where((f.col('_tie_DOD') == 1) & (f.col('RECORD_DEATHS') == 1))\
    .groupBy('PERSON_ID')\
    .agg(\
      f.countDistinct(f.col('DOD')).alias('_n_distinct_DOD')\
      , f.countDistinct(f.when(f.col('DOD').isNull(), 1)).alias('_null_DOD')\
    )\
    .withColumn('_tie_DOD', f.when((f.col(f'_n_distinct_DOD') + f.col(f'_null_DOD')) > 1, 1).otherwise(0))\
    .select('PERSON_ID', '_tie_DOD')
  
  _ties = _ties\
    .join(_tmp_tie_DOD, on=['PERSON_ID'], how='outer')    
  
    
  # ------------------------------------------------------------------------------------
  # characteristic selection
  # ------------------------------------------------------------------------------------       
  # get information from the first row 
  _dob = _unassembled\
    .select('PERSON_ID', 'RECORD_DATE', 'RECORD_SOURCE', 'DOB', '_rownum_DOB')\
    .where(f.col('_rownum_DOB') == 1)\
    .withColumnRenamed('RECORD_DATE', '_date_DOB')\
    .withColumnRenamed('RECORD_SOURCE', '_source_DOB')\
    .select('PERSON_ID', 'DOB', '_date_DOB', '_source_DOB')
  _sex = _unassembled\
    .select('PERSON_ID', 'RECORD_DATE', 'RECORD_SOURCE', 'SEX', '_rownum_SEX')\
    .where(f.col('_rownum_SEX') == 1)\
    .withColumnRenamed('RECORD_DATE', '_date_SEX')\
    .withColumnRenamed('RECORD_SOURCE', '_source_SEX')\
    .select('PERSON_ID', 'SEX', '_date_SEX', '_source_SEX')
  _ethnic = _unassembled\
    .select('PERSON_ID', 'RECORD_DATE', 'RECORD_SOURCE', 'ETHNIC', '_rownum_ETHNIC')\
    .where(f.col('_rownum_ETHNIC') == 1)\
    .withColumnRenamed('RECORD_DATE', '_date_ETHNIC')\
    .withColumnRenamed('RECORD_SOURCE', '_source_ETHNIC')\
    .select('PERSON_ID', 'ETHNIC', '_date_ETHNIC', '_source_ETHNIC')
  _lsoa = _unassembled\
    .select('PERSON_ID', 'RECORD_DATE', 'RECORD_SOURCE', 'LSOA', '_rownum_LSOA')\
    .where(f.col('_rownum_LSOA') == 1)\
    .withColumnRenamed('RECORD_DATE', '_date_LSOA')\
    .withColumnRenamed('RECORD_SOURCE', '_source_LSOA')\
    .select('PERSON_ID', 'LSOA', '_date_LSOA', '_source_LSOA')

  
  # dod is extracted from the unfiltered table
  #   would like to keep RECORD_DEATHS before rownum...
  _dod = _unassembled_orig\
    .withColumn('RECORD_DEATHS', f.when(f.col('RECORD_SOURCE') == 'deaths', 1).otherwise(0))\
    .withColumn('_rownum_DOD', f.row_number().over(_win_rownum_DOD))\
    .where((f.col('_rownum_DOD') == 1) & (f.col('RECORD_DEATHS') == 1))\
    .withColumnRenamed('RECORD_DATE', '_date_DOD')\
    .withColumnRenamed('RECORD_SOURCE', '_source_DOD')\
    .select('PERSON_ID', 'DOD', '_date_DOD', '_source_DOD')
  
  
  # ------------------------------------------------------------------------------------  
  # gdppr presence
  # ------------------------------------------------------------------------------------  
  # add as an option to the function
  # in_gdppr is extracted from the unfiltered table
  #   so in_gdppr may relate to records after the CENSOR_DATE
  _in_gdppr = _unassembled_orig\
    .where(f.col('RECORD_SOURCE') == 'gdppr')\
    .select('PERSON_ID')\
    .distinct()\
    .where(f.col('PERSON_ID').isNotNull())\
    .withColumn('in_gdppr', f.lit(1))
  
  
  # ------------------------------------------------------------------------------------  
  # ethnic categorisaton *** TODO ***
  # ------------------------------------------------------------------------------------    
  #   withColumn('ETHNIC_CAT',\
  #     f.when(f.col('ETHNIC'))
  #   )

  # spark.sql(f"""
  # CREATE OR REPLACE GLOBAL TEMP VIEW {project_prefix}ethnicity_lookup AS
  # SELECT *, 
  #       CASE WHEN ETHNICITY_CODE IN ('1','2','3','N','M','P') THEN "Black or Black British"
  #            WHEN ETHNICITY_CODE IN ('0','A','B','C') THEN "White"
  #            WHEN ETHNICITY_CODE IN ('4','5','6','L','K','J','H') THEN "Asian or Asian British"
  #            WHEN ETHNICITY_CODE IN ('7','8','W','T','S','R') THEN "Other Ethnic Groups"
  #            WHEN ETHNICITY_CODE IN ('D','E','F','G') THEN "Mixed"
  #            WHEN ETHNICITY_CODE IN ('9','Z','X') THEN "Unknown "
  #            ELSE 'Unknown' END as ETHNIC_GROUP  
  # FROM (
  #   SELECT ETHNICITY_CODE, ETHNICITY_DESCRIPTION FROM dss_corporate.hesf_ethnicity
  #   UNION ALL
  #   SELECT Value as ETHNICITY_CODE, Label as ETHNICITY_DESCRIPTION FROM dss_corporate.gdppr_ethnicity WHERE Value not in (SELECT ETHNICITY_CODE FROM FROM dss_corporate.hesf_ethnicity)) """)

  #   _dss_mapping_snomed_ethnic

  
  # ------------------------------------------------------------------------------------  
  # assemble
  # ------------------------------------------------------------------------------------  
  _assembled = _dob\
    .join(_sex, on=['PERSON_ID'], how='outer')\
    .join(_ethnic, on=['PERSON_ID'], how='outer')\
    .join(_lsoa, on=['PERSON_ID'], how='outer')\
    .join(_dod, on=['PERSON_ID'], how='outer')\
    .join(_in_gdppr, on=['PERSON_ID'], how='outer')\
    .join(_ties, on=['PERSON_ID'], how='outer')
  
  if(_individual_censor_dates): 
    _assembled = _assembled\
      .join(_individual_censor_dates, on=['PERSON_ID'], how='inner')
  
  return _assembled

  # Note apply ethnicity lookup afterwards 

# COMMAND ----------

# MAGIC %md ## 5.2 Save Data

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
assembled = skinny_assembled(_unassembled=unassembled, _overall_censor_date='2023-10-31')

if save_data:
  save_dataset('_tmp_skinny_assembled', assembled)
assembled = read_dataset('_tmp_skinny_assembled')

count_var(assembled, 'PERSON_ID')

# COMMAND ----------

for v in list(vars().keys()):
  if(re.search("dataframe", type(vars()[v]).__name__.lower())):
    print(v, type(vars()[v]))

# COMMAND ----------

# MAGIC %md # 6 Ethnic

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

path_cur_lsoa_region       = f'{dbc}.{proj}_cur_lsoa_region_lookup'
path_cur_lsoa_imd          = f'{dbc}.{proj}_cur_lsoa_imd_lookup'
path_cur_ethnic_desc_cat   = f'{dbc}.{proj}_cur_ethnic_desc_cat_lookup'
#path_cur_covid             = f'{dbc}.{proj}_cur_covid'

spark.sql(f'''
REFRESH TABLE {path_cur_ethnic_desc_cat}
''')
spark.sql(f'''
REFRESH TABLE {path_cur_lsoa_region}
''')
spark.sql(f'''
REFRESH TABLE {path_cur_lsoa_imd}
''')
ethnic_desc_cat = spark.table(path_cur_ethnic_desc_cat)
lsoa_region     = spark.table(path_cur_lsoa_region)
lsoa_imd        = spark.table(path_cur_lsoa_imd)

# COMMAND ----------

# check
count_var(ethnic_desc_cat, 'ETHNIC'); print()
count_var(assembled, 'ETHNIC'); print()

# COMMAND ----------

display(assembled.select(f.col("ETHNIC")).groupBy('ETHNIC').count().sort(f.col("ETHNIC").desc()))

# COMMAND ----------

# add ethnicity description and category
# merge (right_only) - equivalent to left join
assembled = merge(assembled, ethnic_desc_cat, ['ETHNIC'])\
  .where(f.col('_merge') != 'right_only')
print()

# check
tmpt = tab(assembled, 'ETHNIC', '_merge', var2_unstyled=1); print()
count_var(assembled, 'PERSON_ID'); print()

# check
tmpt = tab(assembled, 'ETHNIC_DESC', 'ETHNIC_CAT', var2_unstyled=1); print()
tmpt = tab(assembled, 'ETHNIC_CAT'); print()

# edit
assembled = assembled\
  .withColumn('ETHNIC_CAT',\
    f.when(f.col('ETHNIC_CAT').isNull(), 'Unknown')\
     .otherwise(f.col('ETHNIC_CAT'))\
  )

# check
tmpt = tab(assembled, 'ETHNIC_CAT'); print()

# tidy
assembled = assembled.drop('_merge')

# COMMAND ----------

tmpt = tab(assembled, 'ETHNIC', 'ETHNIC_CAT', var2_unstyled=1); print()


# COMMAND ----------

# MAGIC %md # 7 IMD

# COMMAND ----------

# MAGIC %md ## 7.1 LSOA

# COMMAND ----------

# prepare
lsoa_region = lsoa_region\
  .select('LSOA', 'region')

# check
count_var(lsoa_region, 'LSOA'); print()
count_var(assembled, 'LSOA'); print()
tab(lsoa_region, 'region'); print()

# merge
assembled = (merge(assembled, lsoa_region, ['LSOA'])
             .where(f.col('_merge') != 'right_only')
             .withColumn('LSOA_1', f.substring(f.col('LSOA'), 1, 1))
            )
print()

# check
tab(assembled, 'LSOA_1', '_merge', var2_unstyled=1); print()
count_var(assembled, 'PERSON_ID'); print()

# edit
assembled = (assembled
             .withColumn('region',
                         f.when(f.col('LSOA_1') == 'W', 'Wales')
                         .when(f.col('LSOA_1') == 'S', 'Scotland')
                         .otherwise(f.col('region'))
                        )
            )

# check
tab(assembled, 'region'); print()

# tidy
assembled = assembled.drop('_merge', 'LSOA_1')

# COMMAND ----------

# MAGIC %md ## 7.2 Convert LSOA to IMD

# COMMAND ----------

# check
count_var(lsoa_imd, 'LSOA'); print()
count_var(assembled, 'LSOA'); print()
tmpt = tab(lsoa_imd, 'IMD_2019_DECILES', 'IMD_2019_QUINTILES', var2_unstyled=1); print()

# merge
assembled = merge(assembled, lsoa_imd, ['LSOA'])\
  .where(f.col('_merge') != 'right_only')\
  .withColumn('LSOA_1', f.substring(f.col('LSOA'), 1, 1))
print()

# check
tmpt = tab(assembled, 'LSOA_1', '_merge', var2_unstyled=1); print()
count_var(assembled, 'PERSON_ID'); print()

# tidy
assembled = assembled\
  .drop('_merge', 'LSOA_1')

# COMMAND ----------

assembled = assembled.select(['PERSON_ID', 'DOB', 'SEX', 'DOD', 'LSOA', 'ETHNIC', 
 'ETHNIC_DESC', 'ETHNIC_CAT', 'region', 'IMD_2019_DECILES', 'IMD_2019_QUINTILES', 'in_gdppr'])

# COMMAND ----------

display(assembled)

# COMMAND ----------

# MAGIC %md # 8 Save Final Assembled

# COMMAND ----------

# save
proj = 'ccu059_01'
db = 'dars_nic_391419_j3w9t'
dbc = f'{db}_collab'

outName = f'{proj}_final_skinny_assembled'.lower()

assembled.createOrReplaceGlobalTempView(outName)
drop_table(outName, if_exists=True)
create_table(outName, select_sql_script=f"SELECT * FROM global_temp.{outName}")
