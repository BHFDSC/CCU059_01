# Databricks notebook source
# MAGIC %md # CCU059_01_D02_Lookup_Data
# MAGIC
# MAGIC **Description** This notebook produces the curated tables.
# MAGIC
# MAGIC **Original Author(s)** Tom Bolton (John Nolan, Elena Raffetti, CCU002)
# MAGIC
# MAGIC **Data Wrangler for CCU059(s)** Sharmin Shabnam

# COMMAND ----------

spark.sql('CLEAR CACHE')

# COMMAND ----------

# DBTITLE 1,Libraries
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

print("Matplotlib version: ", matplotlib.__version__)
print("Seaborn version: ", sns.__version__)
_datetimenow = datetime.datetime.now() 
print(f"_datetimenow:  {_datetimenow}")

# COMMAND ----------

# MAGIC %run "/Users/ss1279@leicester.ac.uk/Helper_functions/functions"

# COMMAND ----------

#Datasets
# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
proj = 'ccu059_01'


# -----------------------------------------------------------------------------
# Databases
# -----------------------------------------------------------------------------
db = 'dars_nic_391419_j3w9t'
dbc = f'{db}_collab'

# reference tables
path_ref_bhf_phenotypes  = 'bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127'
path_ref_map_ctv3_snomed = 'dss_corporate.read_codes_map_ctv3_to_snomed'
path_ref_geog            = 'dss_corporate.ons_chd_geo_listings'
path_ref_imd             = 'dss_corporate.english_indices_of_dep_v02'
path_ref_ethnic_hes      = 'dss_corporate.hesf_ethnicity'
path_ref_ethnic_gdppr    = 'dss_corporate.gdppr_ethnicity'
path_ref_gp_refset       = 'dss_corporate.gpdata_snomed_refset_full'
path_ref_gdppr_refset    = 'dss_corporate.gdppr_cluster_refset'
path_ref_icd10           = 'dss_corporate.icd10_group_chapter_v01'


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
savedatetime = "20230126"
# in tables (available post table_freeze)
path_deaths        = f'{dbc}.{proj}_freeze_deaths_{savedatetime}'
path_gdppr         = f'{dbc}.{proj}_freeze_gdppr_{savedatetime}'
path_hes_apc       = f'{dbc}.{proj}_freeze_hes_apc_{savedatetime}'
path_chess         = f'{dbc}.{proj}_freeze_chess_{savedatetime}'
path_sgss          = f'{dbc}.{proj}_freeze_sgss_{savedatetime}'
path_sus           = f'{dbc}.{proj}_freeze_sus_{savedatetime}'
path_vacc          = f'{dbc}.{proj}_freeze_vaccine_status_{savedatetime}'

path_codelist      = f'{dbc}.ccu010_01_codelists'

start_date = '2020-01-31'
end_date = '2022-12-31' 
# Need to check the maximum date for which we have coverage for all data sources

# COMMAND ----------

# MAGIC %md # 1 Data

# COMMAND ----------

geog         = spark.table(path_ref_geog)
imd          = spark.table(path_ref_imd)
ethnic_hes   = spark.table(path_ref_ethnic_hes)
ethnic_gdppr = spark.table(path_ref_ethnic_gdppr)

# COMMAND ----------

# MAGIC %md # 4 LSOA region lookup

# COMMAND ----------

# MAGIC %md ## 4.1 Create

# COMMAND ----------

spark.sql(f"""
  CREATE or replace global temporary view {proj}_lsoa_region_lookup AS
  with curren_chd_geo_listings as (
    SELECT * 
    FROM {path_ref_geog}
    --WHERE IS_CURRENT = 1
  ),
  lsoa_auth as (
    SELECT e01.geography_code as lsoa_code, e01.geography_name lsoa_name, 
      e02.geography_code as msoa_code, e02.geography_name as msoa_name, 
      e0789.geography_code as authority_code, e0789.geography_name as authority_name,
      e0789.parent_geography_code as authority_parent_geography
    FROM curren_chd_geo_listings e01
    LEFT JOIN curren_chd_geo_listings e02 on e02.geography_code = e01.parent_geography_code
    LEFT JOIN curren_chd_geo_listings e0789 on e0789.geography_code = e02.parent_geography_code
    WHERE e01.geography_code like 'E01%' and e02.geography_code like 'E02%'
  ),
  auth_county as (
    SELECT lsoa_code, lsoa_name,
           msoa_code, msoa_name,
           authority_code, authority_name,
           e10.geography_code as county_code, e10.geography_name as county_name,
           e10.parent_geography_code as parent_geography
    FROM lsoa_auth
    LEFT JOIN dss_corporate.ons_chd_geo_listings e10 on e10.geography_code = lsoa_auth.authority_parent_geography
    WHERE LEFT(authority_parent_geography,3) = 'E10'
  ),
  auth_met_county as (
    SELECT lsoa_code, lsoa_name,
           msoa_code, msoa_name,
           authority_code, authority_name,
           NULL as county_code, NULL as county_name,           
           lsoa_auth.authority_parent_geography as region_code
    FROM lsoa_auth
    WHERE LEFT(authority_parent_geography,3) = 'E12'
  ),
  lsoa_region_code as (
    SELECT lsoa_code, lsoa_name,
           msoa_code, msoa_name,
           authority_code, authority_name,
           county_code, county_name, 
           auth_county.parent_geography as region_code
    FROM auth_county
    UNION ALL
    SELECT lsoa_code, lsoa_name,
           msoa_code, msoa_name,
           authority_code, authority_name,
           county_code, county_name, 
           region_code 
    FROM auth_met_county
  ),
  lsoa_region as (
    SELECT lsoa_code, lsoa_name,
           msoa_code, msoa_name,
           authority_code, authority_name,
           county_code, county_name, 
           region_code, e12.geography_name as region_name 
    FROM lsoa_region_code
    LEFT JOIN dss_corporate.ons_chd_geo_listings e12 on lsoa_region_code.region_code = e12.geography_code
  )
  SELECT * FROM lsoa_region
""")

# COMMAND ----------

tmp1 = spark.table(f'global_temp.{proj}_lsoa_region_lookup')
display(tmp1)

# COMMAND ----------

# MAGIC %md ## 4.2 Check

# COMMAND ----------

count_var(tmp1, 'lsoa_code')

# COMMAND ----------

# check duplicates
w1 = Window\
  .partitionBy('lsoa_code')\
  .orderBy('lsoa_name')
w2 = Window\
  .partitionBy('lsoa_code')
tmp2 = tmp1\
  .withColumn('_rownum', f.row_number().over(w1))\
  .withColumn('_rownummax', f.count('lsoa_code').over(w2))\
  .where(f.col('_rownummax') > 1)
display(tmp2)
# duplicates are a result of an authority name change - not relevant for this project

# COMMAND ----------

tmp2 = tmp1\
  .withColumn('_rownum', f.row_number().over(w1))\
  .where(f.col('_rownum') == 1)\
  .select('lsoa_code', 'lsoa_name', 'region_code', 'region_name')\
  .withColumnRenamed('lsoa_code', 'LSOA')\
  .withColumnRenamed('region_name', 'region')

count_var(tmp2, 'LSOA'); print()
tmpt = tab(tmp2, 'region'); print()

# COMMAND ----------

# MAGIC %md ## 4.3 Save 

# COMMAND ----------

# save name
# temp save
proj = 'ccu059_01'
db = 'dars_nic_391419_j3w9t'
dbc = f'{db}_collab'
outName = f'{proj}_cur_lsoa_region_lookup'.lower()

tmp2.write.mode('overwrite').saveAsTable(f'{dbc}.{outName}')
spark.sql(f'ALTER TABLE {dbc}.{outName} OWNER TO {dbc}')

# COMMAND ----------

# MAGIC %md # 5 LSOA IMD lookup

# COMMAND ----------

# check
print(imd.toPandas().head(5)); print()
count_var(imd, 'LSOA_CODE_2011'); print()
tmpt = tab(imd, 'DECI_IMD', 'IMD_YEAR', var2_unstyled=1); print()

# tidy
tmp1 = imd\
  .where(f.col('IMD_YEAR') == 2019)\
  .select('LSOA_CODE_2011', 'DECI_IMD')\
  .withColumnRenamed('LSOA_CODE_2011', 'LSOA')\
  .withColumn('IMD_2019_QUINTILES',
    f.when(f.col('DECI_IMD').isin([1,2]), 1)\
     .when(f.col('DECI_IMD').isin([3,4]), 2)\
     .when(f.col('DECI_IMD').isin([5,6]), 3)\
     .when(f.col('DECI_IMD').isin([7,8]), 4)\
     .when(f.col('DECI_IMD').isin([9,10]), 5)\
     .otherwise(None)\
  )\
  .withColumnRenamed('DECI_IMD', 'IMD_2019_DECILES')

# check
tmpt = tab(tmp1, 'IMD_2019_DECILES', 'IMD_2019_QUINTILES', var2_unstyled=1); print()
print(tmp1.toPandas().head(5)); print()

# COMMAND ----------

# MAGIC %md ## 5.3 Save

# COMMAND ----------

# save name
proj = 'ccu059_01'
db = 'dars_nic_391419_j3w9t'
dbc = f'{db}_collab'

outName = f'{proj}_cur_lsoa_imd_lookup'.lower()
tmp1.write.mode('overwrite').saveAsTable(f'{dbc}.{outName}')
spark.sql(f'ALTER TABLE {dbc}.{outName} OWNER TO {dbc}')

# COMMAND ----------

# MAGIC %md # 6 Ethnic desc cat lookup

# COMMAND ----------

display(ethnic_hes)

# COMMAND ----------

print('------------------------------------------------------------------------------')
print('ethnic_hes')
print('------------------------------------------------------------------------------')
# check
print(ethnic_hes.toPandas().head(5)); print()

# prepare
tmp1 = ethnic_hes\
  .select('ETHNICITY_CODE', 'ETHNICITY_DESCRIPTION')\
  .withColumnRenamed('ETHNICITY_CODE', 'ETHNIC')\
  .withColumnRenamed('ETHNICITY_DESCRIPTION', 'ETHNIC_DESC_HES')

# check
count_var(tmp1, 'ETHNIC'); print()
print(tmp1.toPandas().head(5)); print()


print('------------------------------------------------------------------------------')
print('ethnic_gdppr')
print('------------------------------------------------------------------------------')
# check
print(ethnic_gdppr.toPandas().head(5)); print()

# prepare
tmp2 = ethnic_gdppr\
  .select('Value', 'Label')\
  .withColumnRenamed('Value', 'ETHNIC')\
  .withColumnRenamed('Label', 'ETHNIC_DESC_GDPPR')

# check
count_var(tmp2, 'ETHNIC'); print()
print(tmp2.toPandas().head(5)); print()


print('------------------------------------------------------------------------------')
print('merge')
print('------------------------------------------------------------------------------')
# merge
tmp3 = merge(tmp1, tmp2, ['ETHNIC']); print()
tmp3 = (tmp3
        .withColumn('ETHNIC_DESC', f.coalesce(f.col('ETHNIC_DESC_HES'), f.col('ETHNIC_DESC_GDPPR')))
        .orderBy('ETHNIC')
        .withColumn('ETHNIC_DESC', f.when(f.col('_merge') == 'both',
                                          f.col('ETHNIC_DESC_GDPPR')).otherwise(f.col('ETHNIC_DESC')))
        .withColumn('ETHNIC_DESCx', f.concat(f.col('ETHNIC'), f.lit(' '), f.col('ETHNIC_DESC')))
       )

# check
# with pd.option_context('expand_frame_repr', False):
print(tmp3.toPandas().to_string()); print()
count_var(tmp3, 'ETHNIC'); print()

# tidy and add ETHNIC_CAT (CCU002_01-D04)
tmp4 = tmp3\
  .select('ETHNIC', 'ETHNIC_DESC')\
  .withColumn('ETHNIC_CAT',\
   f.when(f.col('ETHNIC').isin(['0','A','B','C']), f.lit('White'))\
    .when(f.col('ETHNIC').isin(['1','2','3','N','M','P']), f.lit('Black'))\
    .when(f.col('ETHNIC').isin(['4','5','6','K','J','H']), f.lit('South Asian'))\
    .when(f.col('ETHNIC').isin(['7','8','W','T','S','R', 'L', 'D','E','F','G']), f.lit('Mixed/Other'))\
    .when(f.col('ETHNIC').isin(['9','Z','X']), f.lit('Unknown'))\
    .otherwise('Unknown'))

# check
tmpt = tab(tmp4, 'ETHNIC_DESC', 'ETHNIC_CAT', var2_unstyled=1); print()

# COMMAND ----------

# MAGIC %md ## 6.2 Save

# COMMAND ----------

# save name
proj = 'ccu059_01'
db = 'dars_nic_391419_j3w9t'
dbc = f'{db}_collab'
# save name
outName = f'{proj}_cur_ethnic_desc_cat_lookup'.lower()
tmp4.write.mode('overwrite').saveAsTable(f'{dbc}.{outName}')
spark.sql(f'ALTER TABLE {dbc}.{outName} OWNER TO {dbc}')

# COMMAND ----------

display(tmp4)
