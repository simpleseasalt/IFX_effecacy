# | ---------------------------------------
# | Author: Simplezzz
# | Date: 2024-08-04 14:41:08
# | LastEditTime: 2024-08-04 15:26:23
# | FilePath: \script\1_data_cleaning.R
# | Description:
# | ---------------------------------------

library(tidyverse)
library(vroom)
library(naniar)

# 1---------------------------------------- load data

data_raw <- vroom("data/tidydata_2023.11.8.csv") %>%
    mutate(
        combine_1 = as.factor(combine_1),
        combine_2 = as.factor(combine_2),
        combine_3 = as.factor(combine_3),
        ADA = as.factor(ADA),
        gender = as.factor(gender),
        validaty = as.factor(validaty)
    ) %>%
    select(monitoring_c, ADA, gender, age, weight, dose, CDAI_before, combine_1, combine_3, CREA, ALT, AST, ALB, WBC, RBC, CDAI_change) %>%
    tidylog::filter(CDAI_before >= 70) %>%
    mutate(
        validaty = if_else(CDAI_change > 70, "yes", "no"),
        validaty = as.factor(validaty)
    )

# 1---------------------------------------- handle missing data
# 2---------------------------------------- summary missing data

data_raw %>%
    miss_var_summary() %>%
    mutate(
        Imputation = case_when(
            pct_miss == 0 ~ "NA",
            pct_miss >= 5 ~ "Multiple imputation",
            variable == "gender" ~ "Majority",
            TRUE ~ "Median"
        )
    ) %>%
    write_csv("D:/OneDrive/After_work/2023/英夫利西单抗/output/1_missing_data.csv")

# 2---------------------------------------- simply imputation with median or mojority

var_simple_imputation <- data_raw %>%
    miss_var_summary() %>%
    filter(pct_miss > 0 & pct_miss < 5) %>%
    pull(variable)

data_simply_imputation <- data_raw %>%
    replace_na(
        list(
            CREA = median(.$CREA, na.rm = TRUE),
            WBC = median(.$WBC, na.rm = TRUE),
            RBC = median(.$RBC, na.rm = TRUE),
            age = median(.$age, na.rm = TRUE),
            gender = "1"
        )
    ) %>%
    mutate(patient = as.factor(1:nrow(.)))

# 2---------------------------------------- multiple imputation

library(mice)

impute_strategy <- mice(data_simply_imputation, m = 5, maxit = 4, method = "rf", seed = 2024)

data_complete_1 <- complete(impute_strategy, action = 1) %>%
    mutate(imputation = 1)

data_complete_2 <- complete(impute_strategy, action = 2) %>%
    mutate(imputation = 2)

data_complete_3 <- complete(impute_strategy, action = 3) %>%
    mutate(imputation = 3)

data_complete_4 <- complete(impute_strategy, action = 4) %>%
    mutate(imputation = 4)

data_complete_5 <- complete(impute_strategy, action = 5) %>%
    mutate(imputation = 5)

data_imputed <- bind_rows(data_complete_1, data_complete_2, data_complete_3, data_complete_4, data_complete_5) %>%
    group_by(patient) %>%
    summarise(
        across(where(is.numeric), mean, na.rm = TRUE),
        across(where(is.factor), ~ factor(levels(.)[which.max(table(.))]))
    ) %>%
    relocate(
        Age = age,
        Gender = gender,
        Weight = weight,
        WBC = WBC,
        RBC = RBC,
        ALT = ALT,
        AST = AST,
        ALB = ALB,
        CREA = CREA,
        ADA = ADA,
        CDAI_before = CDAI_before,
        CDAI_change = CDAI_change
    ) %>%
    relocate(validaty, .after = last_col()) %>%
    select(-c(patient, imputation))

save(data_imputed, file = "output/1_data_imputed.RData")

# ! end