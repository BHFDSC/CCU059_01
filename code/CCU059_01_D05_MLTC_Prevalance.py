# Databricks notebook source
# MAGIC %md # 1 Import Functions
# MAGIC
# MAGIC **Data Wrangler for CCU059(s)** Sharmin Shabnam

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window
from pyspark.sql.window import Window
from pyspark.sql import Row
from functools import reduce
import databricks.koalas as ks
import pandas as pd
import numpy as np
import re
import io
import datetime
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns

_datetimenow = datetime.datetime.now() # .strftime("%Y%m%d")
print(f"_datetimenow:  {_datetimenow}")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.sql('CLEAR CACHE')

# COMMAND ----------

# MAGIC %run "/Repos/ss1279@leicester.ac.uk/ccu059/Helper_functions/functions"

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
path_cohort_2021   = f'{dbc}.{proj}_tmp_cohort_2021'
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
  #selected.remove('cond_total')
  #selected.remove('cond_concat')
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

spark.sql(F"""REFRESH TABLE {path_cohort_2021}""")
spark.sql(F"""REFRESH TABLE {path_skinny}""")

cohort_2021        = spark.table(path_cohort_2021)
skinny_data        = spark.table(path_skinny)
deaths             = spark.table(path_deaths)
hes_apc            = spark.table(path_hes_apc)
gdppr              = spark.table(path_gdppr)

display(cohort_2021)

# COMMAND ----------

count_var(cohort_2021, 'person_id')

# COMMAND ----------

# MAGIC %md # 3 Codelist

# COMMAND ----------

spark.sql(F"""REFRESH TABLE {path_codelist}""")
codelist = spark.table(path_codelist)
tab(codelist, 'name' , 'terminology', var2_unstyled=1); print()

# COMMAND ----------

mltc_names = codelist.select(f.col("cond")).orderBy('cond').toPandas()["cond"].unique().tolist() 
print(mltc_names)

# COMMAND ----------

codelist_gdppr = (codelist
                  .filter(f.col('terminology')=='SNOMED') 
                  .withColumnRenamed('code', 'code')
                  .withColumnRenamed('name', 'cond_name')
                  .select(['cond', 'code', 'cond_name'])
                 )

codelist_hes = (codelist
                  .filter(f.col('terminology')=='ICD10') 
                  .withColumnRenamed('code', 'code')
                  .withColumnRenamed('name', 'cond_name')
                  .select(['cond', 'code', 'cond_name'])
                 )display(codelist_hes)

# COMMAND ----------

# MAGIC %md # 4 Prepare datasets

# COMMAND ----------

# MAGIC %md ## 4.0 Load Skinny Table

# COMMAND ----------

_patient = (skinny_data
            .select(['PERSON_ID', 'DOB', 'DOD', 'SEX'])
            .withColumn('censor_start_date', f.date_add(f.col('DOB'), -0))
            .where(f.col('PERSON_ID').isNotNull() & f.col('DOB').isNotNull() & f.col('SEX').isNotNull())
            .withColumn('censor_end_date',
                        f.when(
                          (f.col('DOD').isNotNull()),
                          f.col('DOD'))
                        .otherwise(f.to_date(f.lit('2023-03-31')))
                     )
            .filter((f.col("censor_end_date") > '2020-12-01'))
            .select(['PERSON_ID', 'censor_start_date', 'censor_end_date'])
            .withColumnRenamed('PERSON_ID', 'person_id')
            .orderBy('censor_start_date')
              )
display(_patient)

# COMMAND ----------

# MAGIC %md ## 4.1 GDPPR

# COMMAND ----------

# ==========================================
#              GDPPR
# ==========================================
_gdppr = (gdppr
          .limit(1000)
          .select(['NHS_NUMBER_DEID', 'DATE', 'CODE'])
          .withColumnRenamed('NHS_NUMBER_DEID', 'PERSON_ID')
          .withColumnRenamed('CODE', 'code')
          .withColumnRenamed('DATE', 'date')
          .join(_patient, ['PERSON_ID'], 'left')
          .where((f.col('date') >= f.col('censor_start_date')) & (f.col('date') <= f.col('censor_end_date')))
              )
display(_gdppr)

_gdppr_codes = (gdppr
                .select(['NHS_NUMBER_DEID', 'DATE', 'CODE'])
                .withColumnRenamed('NHS_NUMBER_DEID', 'person_id')
                .withColumnRenamed('CODE', 'code')
                .withColumnRenamed('DATE', 'date')
                .where(f.col('person_id').isNotNull() & f.col('date').isNotNull())
                .distinct()
                .orderBy(['person_id', 'date'], ascending = True)
                .join(codelist_gdppr, ['code'], 'leftsemi')
                .select(['person_id', 'date', 'code'])
               )
_gdppr_codes = (_patient
                 .join(_gdppr_codes, ['person_id'], 'left')
                 .where((f.col('date') >= f.col('censor_start_date')) 
                        & (f.col('date') <= f.col('censor_end_date')))
               )
save_dataset('_tmp_gdppr_relevant_snomed_codes', _gdppr_codes)
_gdppr_codes = read_dataset('_tmp_gdppr_relevant_snomed_codes')
count_var(_gdppr_codes, 'person_id')

_gdppr_codes = (_gdppr_codes
                .select(['person_id', 'censor_start_date', 'censor_end_date', 'date', 'code'])
                .join(codelist_gdppr.select(['code', 'cond']), ['code'], 'left')
                .select(['person_id', 'censor_start_date', 'censor_end_date', 'date', 'code', 'cond'])
               )
count_var(_gdppr_codes, 'person_id')

# COMMAND ----------

_gdppr_codes.select(['code', 'cond']).groupBy(['code', 'cond']).count().sort(f.col('count').desc()).show()

# COMMAND ----------

# MAGIC %md ## 4.2 HES APC

# COMMAND ----------

codelist_hes = codelist_hes.select(['code', 'cond'])
codelist_hes_dict = {row['code']:row['cond'] for row in codelist_hes.collect()}

codelist_hes_dict_keys = list(codelist_hes_dict.keys())
codelist_hes_dict_keys = '|'.join(codelist_hes_dict_keys)
print(codelist_hes_dict_keys)

def get_code_list(codelist): 
  code_list = (codelist.select(f.col("code")).toPandas()["code"])
  code_list = list(map(lambda x: str(x), code_list))
  return code_list

codelist_hes_list = get_code_list(codelist_hes)

# COMMAND ----------

_hes_apc = (hes_apc
            .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')
            .select(['PERSON_ID', 'EPIKEY', 'EPISTART'] 
                    + [col for col in list(hes_apc.columns) if re.match(r'^DIAG_(3|4)_\d\d$', col)])
            .orderBy('PERSON_ID', 'EPIKEY')  
           )

_hes_apc_codes = (reshape_wide_to_long_multi(_hes_apc,
                                           i=['PERSON_ID', 'EPIKEY', 'EPISTART'],
                                           j='POSITION', stubnames=['DIAG_4_', 'DIAG_3_'])
               )
_hes_apc_codes = (reshape_wide_to_long_multi(_hes_apc_codes,
                                           i=['PERSON_ID', 'EPIKEY', 'EPISTART', 'POSITION'],
                                           j='DIAG_DIGITS', stubnames=['DIAG_'])
                .withColumnRenamed('POSITION', 'DIAG_POSITION')
                .withColumn('DIAG_POSITION', f.regexp_replace('DIAG_POSITION', r'^[0]', ''))
                .withColumn('DIAG_DIGITS', f.regexp_replace('DIAG_DIGITS', r'[_]', ''))
                .withColumn('DIAG_', f.regexp_replace('DIAG_', r'X$', ''))
                .withColumn('DIAG_', f.regexp_replace('DIAG_', r'[.,\-\s]', ''))
                .withColumnRenamed('DIAG_', 'CODE')
                .where((f.col('CODE').isNotNull()) & (f.col('CODE') != ''))
                .orderBy(['PERSON_ID', 'EPIKEY', 'DIAG_DIGITS', 'DIAG_POSITION'])
                .filter(f.col('DIAG_DIGITS')==4)
                .withColumnRenamed('EPISTART', 'date')
                .withColumnRenamed('DIAG_DIGITS', 'diag_digits')
                .withColumnRenamed('CODE', 'code')
                .select(['PERSON_ID', 'date', 'code', 'diag_digits'])
                .distinct()
               )
_hes_apc_codes = (_hes_apc_codes
                  .where(f.col('code').rlike(codelist_hes_dict_keys))
)
save_dataset('_tmp_hes_long_codes_20230428', _hes_apc_codes)
_hes_apc_codes = read_dataset('_tmp_hes_long_codes_20230428')
count_var(_hes_apc_codes, 'PERSON_ID')

# COMMAND ----------

_hes_apc_codes = (_hes_apc_codes
                  .withColumn('cond', f.regexp_extract('code', codelist_hes_dict_keys, 0))
                  .replace(codelist_hes_dict, subset=['cond'])
                  .select(['PERSON_ID', 'code', 'date', 'cond'])
                  .withColumnRenamed('PERSON_ID','person_id')
                  .distinct()
                  )

_hes_apc_codes = (_patient
                 .join(_hes_apc_codes, ['person_id'], 'left')
                 .where((f.col('date') >= f.col('censor_start_date')) 
                        & (f.col('date') <= f.col('censor_end_date')))
                  .orderBy(['person_id', 'date', 'cond'])
                  .select(['person_id', 'censor_start_date', 'censor_end_date', 'date', 'code', 'cond'])
            )

# COMMAND ----------

# MAGIC %md # 5 Combine GDPPR HES

# COMMAND ----------

_patient_mltc = (
  _hes_apc_codes
  .unionByName(_gdppr_codes)
  .orderBy(['person_id', 'date', 'cond'])
  )

_win = Window.partitionBy(['person_id', 'cond']).orderBy('date')  
_patient_mltc = (_patient_mltc
    .withColumn('_rownum', f.row_number().over(_win))
    .where(f.col('_rownum') == 1)
    .select('person_id', 'censor_start_date', 'censor_end_date', 'date', 'code', 'cond')
    .orderBy('person_id', 'date', 'cond')
               )

# COMMAND ----------

# MAGIC %md ## 5.1 Save

# COMMAND ----------

_name_prefix='cond_'
_patient_mltc_wide = (_patient_mltc
                       .withColumn('cond', f.concat(f.lit(f'{_name_prefix}'), f.lower(f.col('cond'))))
    .groupBy('person_id', 'censor_start_date', 'censor_end_date')
    .pivot('cond')
    .agg(f.first('date'))
    .where(f.col('person_id').isNotNull())
    .orderBy('person_id') 
                      )
save_dataset('_tmp_mltc_cohort_20230428', _patient_mltc_wide)
_patient_mltc_wide = read_dataset('_tmp_mltc_cohort_20230428').drop('censor_start_date', 'censor_end_date')

# COMMAND ----------

print_mm_summary(_patient_mltc_wide)

# COMMAND ----------

# MAGIC %md # 6 Cohort MLTC 

# COMMAND ----------

_cohort_2021 = read_dataset('_tmp_cohort_2021')
count_var(_cohort_2021, 'person_id')

# COMMAND ----------

display(_cohort_2021)

# COMMAND ----------

mltc_names = codelist.select(f.col("cond")).orderBy('cond').toPandas()["cond"].unique().tolist() 
print(mltc_names)
new_mltc_names = ['axdp', 'asth', 'copd', 'canc', 'ckd', 'chd', 'dem', 'diab', 'hf', 'hyp', 'livd', 'pvd', 'scbp', 'strk', 'epi', 'ibd', 'ms', 'oa', 'op', 'pd', 'td']
cond_vars = ['cond_'+s for s in new_mltc_names]
print(cond_vars)

# COMMAND ----------

def create_save_mltc_cohort(_cohort_x, year, save=False):
  _cohort_x = _cohort_x.join(_patient_mltc_wide, on='person_id', how='left')
  for var in cond_vars:
    _cohort_x = (_cohort_x
                    .withColumn(var,
                                f.when(f.col(var) > f.col('start_date'), f.lit(None)).otherwise(f.col(var)))
                   )
  _cohort_x = (_cohort_x
                  .withColumn("cond_total",
                              sum(_cohort_x[c].isNotNull().cast('int') for c in cond_vars))
                  .withColumn("cond_concat", 
                              f.concat_ws('+',*[f.when(f.col(c).isNotNull(),
                                                     f.split(f.lit(c),'_')[1]) for c in cond_vars]))
                  .withColumn('cond_concat',
                                f.when((f.col('cond_concat')==''),f.lit('none'))
                                .otherwise(f.col('cond_concat')))
                 )
  return _cohort_x

# COMMAND ----------

_cohort_2021 = create_save_mltc_cohort(cohort_2021, 2021, True)
count_var(_cohort_2021, 'person_id')

# COMMAND ----------

_cohort_2021.columns
new_mltc_names = ['axdp', 'asth', 'copd', 'canc', 'ckd', 'chd', 'dem', 'diab', 'hf', 'hyp', 'livd', 'pvd', 'scbp', 'strk', 'epi', 'ibd', 'ms', 'oa', 'op', 'pd', 'td']

cond_vars = ['cond_'+s for s in new_mltc_names]
print(cond_vars)

# COMMAND ----------

save_dataset('_tmp_cohort_2021_mltc', _cohort_2021)
cohort_2021 = read_dataset('_tmp_cohort_2021_mltc')
count_var(_cohort_2021, 'person_id')

# COMMAND ----------

# MAGIC %md # 7 Prevalance output

# COMMAND ----------

def print_mm_summary(df, start_date):
  a=[];b=[];c=[];d=[]
  selected = [s for s in df.columns if 'cond_' in s]
  selected.remove('cond_total')
  selected.remove('cond_concat')
  for var in selected:
    a.append(var)
    b.append(df.select(f.col(var)).filter(df[var].isNotNull()).count())
    c.append(df.count())
    d.append(df
             .select([var, 'start_date'])
             .withColumn("time", f.datediff(f.col("start_date"),f.col(var))/365.25)
             .select(['time'])
             .agg({"time": "avg"})
             .collect()[0][0]
            )
  mm_summary = pd.DataFrame({
    'comorbidity': a,
    'patient_count': b,
    'total_patient_count': c,
    'mean_duration_in_years': d,
      })
  def custom_round(x, base=5):
    return int(base * round(float(x)/base))
  mm_summary['patient_count'] = mm_summary['patient_count'].apply(lambda x: custom_round(x, base=5))
  mm_summary['patient_count'] = mm_summary['patient_count'].astype(int)
  mm_summary['total_patient_count'] = mm_summary['total_patient_count'].apply(lambda x: custom_round(x, base=5))
  mm_summary['total_patient_count'] = mm_summary['total_patient_count'].astype(int)
  mm_summary['mean_duration_in_years'] = mm_summary['mean_duration_in_years'].apply(lambda x: round(x, 1))
  mm_summary.loc[0, 'Context'] = f'This table is describing the prevalance of each of the comorbity in our cohort. The first column is the code name of the comorbidity, the second column is the total number of patients with that comorbidity, the third column is the number of total patients in the dataset, the fourth column is the mean duration (in years) of the comorbidity at study start date {start_date}'
  schema = t.StructType([t.StructField('comorbidity', t.StringType(), True),
                           t.StructField('patient_count', t.IntegerType(), True),
                           t.StructField('total_patient_count', t.IntegerType(), True),
                           t.StructField('mean_duration_in_years', t.FloatType(), True),
                           t.StructField('Context', t.StringType(), True)]
                       )
  df = spark.createDataFrame(mm_summary,schema=schema)
  return df


# COMMAND ----------

df = print_mm_summary(_cohort_2021, '2021-12-01')
display(df)
