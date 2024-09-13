#		Project: NHS Winter Pressure	****
# 		Data curation, linkage, variable definition were conducted in Python (see separate scripts)
#		Data analysis was conducted in R (see below)

#	*** 	1.0: Load required R packages (including those required for initial connection setup

library('DBI')
library('odbc')
library('tidyverse')
library('dplyr')
#library('data.table')
#library('ggplot2')

#if(!require("sandwich")) install.packages("sandwich")
library(sandwich)
if(!require("broom")) install.packages("broom")
library(broom)
if(!require("lmtest")) install.packages("lmtest")
library(lmtest)


#	***		1.1: environment and directory set up
rm(list = ls())
memory.limit(size=50000)
set.seed(12345)
options(scipen = 50)

#getwd()
# setup the databricks environment and ODBC connection (requires individual password; omitted for privacy reasons
setwd("directory") # specify the directory


# ****		1.2: Descriptive statistics		*********

mltc_ss <- readRDS("mltc_ss.rds")

# for table 1, creating three MLTC categories: 0 (none), 1, and >1
mltc_ss <- mltc_ss %>%
  mutate(mltc = ifelse(cond_total > 1, 2,cond_total))

# define follow up time
mltc_ss <- mltc_ss %>%
  mutate(followup_d = as.numeric(followup_end_date - followup_start_date))

gc()

summary(mltc_ss$followup_d)

tabulate(mltc_ss$cond_total)

# total number of hospitalisations or deaths
mltc_ss %>%
  summarise(num_hosp_adm = sum(num_hosp_adm),
            deaths = sum(deaths))


mltc_ss %>%
  group_by(mltc) %>%
  tally()

# total and median follow up time
mltc_ss %>%
  summarise(followup_yrs = sum(followup_d)/365,
            median_folup = median(followup_d))

# Number of patients in each MLTC group
table1_n <- mltc_ss %>%
  group_by(mltc) %>%
  tally() %>%
  mutate(n = round(n/5)*5) %>% # rounded to nearest five as per data privacy rules.
  pivot_wider(names_from = "mltc",
              values_from = "n",
              names_prefix = "mltc_")

# meand (SD) of age
table1_age <- mltc_ss %>%
  group_by(mltc) %>%
  summarise(mean = round(mean(age_at_start),1),
            sd = round(sd(age_at_start), 1)) %>%
  ungroup() %>%
  mutate(n_p = paste0(mean, " (", sd,")")) %>%
  select(!c(mean,sd)) %>%
  mutate(var = "age") %>%
  pivot_wider(names_from = "mltc",
              values_from = "n_p",
              names_prefix = "mltc_")

# sex distribution; n (%)
table1_female <- mltc_ss %>%
  group_by(mltc, sex_female) %>%
  tally() %>%
  mutate(p = round(n/sum(n)*100,1)) %>%
  ungroup() %>%
  mutate(n = round(n/5)*5) %>% # rounded to nearest five as per privacy policy
  mutate(n_p = paste0(n, " (", p,")")) %>%
  select(!c(n,p)) %>%
  mutate(var = "sex_female") %>%
  pivot_wider(names_from = "mltc",
              values_from = "n_p",
              names_prefix = "mltc_") %>%
  rename(var_level = sex_female)

#  ethnicity distribution; n (%)
table1_eth <- mltc_ss %>%
  group_by(mltc, ethnic_cat) %>%
  tally() %>%
  mutate(p = round(n/sum(n)*100,1)) %>%
  ungroup() %>%
  mutate(n = round(n/5)*5) %>%  # rounded to nearest five as per privacy policy
  mutate(n_p = paste0(n, " (", p,")")) %>%
  select(!c(n,p))%>%
  mutate(var = "ethnic_cat") %>%
  pivot_wider(names_from = "mltc",
              values_from = "n_p",
              names_prefix = "mltc_") %>%
  rename(var_level = ethnic_cat)

#  IMD distribution; n (%)

table1_imd <- mltc_ss %>%
  group_by(mltc, imd) %>%
  tally() %>%
  mutate(p = round(n/sum(n)*100,1)) %>%
  ungroup() %>%
  mutate(n = round(n/5)*5) %>% # rounded to nearest five as per privacy policy
  mutate(n_p = paste0(n, " (", p,")")) %>%
  select(!c(n,p))%>%
  mutate(var = "imd") %>%
  pivot_wider(names_from = "mltc",
              values_from = "n_p",
              names_prefix = "mltc_") %>%
  rename(var_level = imd)

table1_combined_full <-   data.table::rbindlist(list(table1_n, table1_age, table1_female, table1_eth, table1_imd),
use.names = TRUE, fill = TRUE)



# ****		1.3: Regression	analysis- hospitalisaion	*********

# load the cleaned datasets
load("winterpress_rImage_21mar.RData")

gc()

# keep top ten MLTC combinations with the highest rates of hospitalisaion
mltc_select_hosp_top10 <-  subset_hosp %>%
  slice_max(ir_hosp_1000, n=10) %>% #
  bind_rows(subset_hosp %>%
              filter(str_detect(cond_concat, 'one'))) # 
# also keeping the hospitalisaion rate for those with one long-term conditions for comparison
       
# subset the data by the selected MLTC combinations
# Then, produce aggregate data by age, sex, ethnicity and IMD (index of multiple deprivation) quintiles
mltc_select_hosp_agg_age_small <- mltc_select_hosp_top10 %>%
               select(cond_concat, rank) %>%
  left_join(., mltc_select_hosp, 
            by = "cond_concat") %>%
  mutate(age_cat = cut(age_at_start, breaks=c(-Inf,seq(30,80,10), Inf), right = F)) %>%
  group_by(rank, cond_concat, age_cat, sex_female, ethnic_cat, imd) %>%
  dplyr::summarise(num_hosp_adm = sum(num_hosp_adm),
                   deaths = sum(deaths),
                   followup_yrs = sum(followup_d) /365) %>%
  ungroup() %>%
  mutate(ir_hosp_1000 = num_hosp_adm / followup_yrs *1000,
         ir_deaths_1000 = deaths / followup_yrs *1000)

table(mltc_select_hosp_agg_age_small$cond_concat)

gc()

# poisson regression (overdispersed); crude
pois_agg_hosp <- glm(as.numeric(num_hosp_adm) ~ relevel(as.factor(cond_concat), ref = "none") + 
                       offset((log(followup_yrs))), 
                     family = quasipoisson, 
                     data = mltc_select_hosp_agg_age_small)

# tidy output
pois_agg_hosp_output <- tidy(pois_agg_hosp, conf.int=TRUE,
                             exponentiate=TRUE) %>% 
  select(term, estimate, conf.low, conf.high) 

# poisson regression (overdispersed); adjusted

pois_agg_hosp_adj <- glm(as.numeric(num_hosp_adm) ~ relevel(as.factor(cond_concat), ref = "none") + 
                           factor(age_cat) + factor(sex_female) + 
                           factor(ethnic_cat) + factor(imd) + 
                           offset((log(followup_yrs))), 
                         family = quasipoisson, 
                         data = mltc_select_hosp_agg_age_small)
# tidy output
pois_agg_hosp_adj_output <- tidy(pois_agg_hosp_adj, conf.int=TRUE,
                                 exponentiate=TRUE) %>% 
  select(term, estimate, conf.low, conf.high) 

gc()



# ****		1.4: Regression	analysis- deaths	*********

# keep top ten MLTC combinations with the highest rates of death

mltc_select_deaths_top10 <-  subset_deaths %>%
  slice_max(ir_deaths_1000, n=10) %>%
  bind_rows(subset_hosp %>%
              filter(str_detect(cond_concat, 'one'))) 
# also keeping the mortality rate for those with one long-term conditions for comparison

# subset the data by the selected MLTC combinations
# Then, produce aggregate data by age, sex, ethnicity and IMD (index of multiple deprivation) quintiles
mltc_select_deaths_agg_age_small <- mltc_select_deaths %>%
  right_join(., mltc_select_deaths_top10 %>%
               select(cond_concat),
             by = "cond_concat") %>% 
  mutate(age_cat = cut(age_at_start, breaks=c(-Inf,seq(30,80,10), Inf), right = F)) %>% #
  group_by(cond_concat, age_cat, sex_female, ethnic_cat, imd) %>%
  dplyr::summarise(num_hosp_adm = sum(num_hosp_adm),
                   deaths = sum(deaths),
                   followup_yrs = sum(followup_d) /365) %>%
  ungroup() %>%
  mutate(ir_hosp_1000 = num_hosp_adm / followup_yrs *1000,
         ir_deaths_1000 = deaths / followup_yrs *1000)

table(mltc_select_deaths_agg_age_small$cond_concat)

# Crude Poisson regression
pois_agg_topdeaths <- glm(as.numeric(deaths) ~ relevel(as.factor(cond_concat), ref = "none") + 
                            offset((log(followup_yrs))), 
                          family = quasipoisson, 
                          data = mltc_select_deaths_agg_age_small)


pois_agg_topdeaths_output <- tidy(pois_agg_topdeaths, conf.int=TRUE,
                                  exponentiate=TRUE) %>% 
  select(term, estimate, conf.low, conf.high) 

#adjusted Poisson regression
pois_agg_adj_topdeaths <- glm(as.numeric(deaths) ~ relevel(as.factor(cond_concat), ref = "none") + 
                                factor(age_cat) + factor(sex_female) + 
                                factor(ethnic_cat) + factor(imd) + 
                                offset((log(followup_yrs))), 
                              family = quasipoisson, 
                              data = mltc_select_deaths_agg_age_small)
# tidy output
pois_agg_adj_topdeaths_output <- tidy(pois_agg_adj_topdeaths, conf.int=TRUE,
                                      exponentiate=TRUE) %>% 
  select(term, estimate, conf.low, conf.high) 



#save.image("winterpress_rImage_21mar_revised.RData")

#load("winterpress_rImage_21mar_revised.RData")

