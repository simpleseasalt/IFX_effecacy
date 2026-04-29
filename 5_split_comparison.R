# | ---------------------------------------
# | Author: Simplezzz
# | Date: 2025-10-19 21:44:09
# | LastEditTime: 2025-10-19 21:55:37
# | FilePath: \R_scripts\7_split_comparison.R
# | Description: 
# | ---------------------------------------

library(tidyverse)
library(gtsummary)
library(rstatix)
library(nortest)

load("output/testset.RData")
load("output/trainset.RData")

IFX_train <- IFX_train %>%
    rename("response" = "group") %>%
    mutate(
        group = "train",
        group = as.factor(group)
    )

IFX_test <- IFX_validation %>%
    rename("response" = "group") %>%
    mutate(
        group = "test",
        group = as.factor(group)
    )

data_total <- IFX_train %>%
    bind_rows(IFX_test)

# ---------------------------------------- 

numeric_var <- data_total %>%
    select_if(is.numeric) %>%
    names()

### ---------------------------------------- normality test

ad.test.multi <- function(x) {
    ad.test(x) %>%
        broom::tidy()
}

normality <- map_df(
    data_total[numeric_var],
    ad.test.multi
) %>%
    bind_cols(variable = numeric_var) %>%
    select(-method) %>%
    rename("p_norm" = "p.value")

### ---------------------------------------- variance test

variance <- map_df(
    data_total[numeric_var],
    function(x) {
        levene_test(data_total, x ~ data_total$group)
    }
) %>%
    select(p_vari = p) %>%
    cbind(variable = numeric_var)

### ---------------------------------------- t test

t_test_res <- data_total %>%
    select(all_of(numeric_var), group) %>%
    pivot_longer(
        cols = all_of(numeric_var),
        names_to = "variable"
    ) %>%
    group_by(variable) %>%
    t_test(value ~ group, var.equal = TRUE) %>%
    select(variable, p_t_test = p)

### ---------------------------------------- wilcox test

wilcox_test_res <- data_total %>%
    select(all_of(numeric_var), group) %>%
    pivot_longer(
        cols = all_of(numeric_var),
        names_to = "variable"
    ) %>%
    group_by(variable) %>%
    wilcox_test(value ~ group) %>%
    select(variable, p_wilcox_test = p)

p_numb <- normality %>%
    left_join(variance, by = "variable") %>%
    left_join(t_test_res, by = "variable") %>%
    left_join(wilcox_test_res, by = "variable") %>%
    group_by(variable) %>%
    mutate(p_final = if_else(p_norm >= 0.05 & p_vari >= 0.05, p_t_test, p_wilcox_test)) %>%
    add_significance(p.col = "p_final")

## ---------------------------------------- nominal variables

nominal_var <- data_total %>%
    select_if(is.factor) %>%
    select(-group) %>%
    names()

p_chisq <- data_total %>%
    select(all_of(nominal_var), group) %>%
    pivot_longer(
        cols = -group,
        names_to = "variable"
    ) %>%
    group_by(variable) %>%
    do(chisq_test(.$group, .$value)) %>%
    select(variable, p_chisq = p)

## ---------------------------------------- fisher

freq <- freq_table(data_total, group, all_of(nominal_var))

fisher_map <- function(x) {
    temp <- freq_table(data_total, group, x) %>%
        select(-prop) %>%
        pivot_wider(
            names_from = "group",
            values_from = "n"
        ) %>%
        replace(is.na(.), 0)
    p_fisher <- temp %>%
        select(all_of(unique(data_total$group))) %>%
        fisher_test(simulate.p.value = TRUE)
    min_n <- temp %>%
        pivot_longer(
            cols = all_of(unique(data_total$group))
        ) %>%
        arrange(value) %>%
        slice(1) %>%
        select(value) %>%
        cbind(variable = x)
    cbind(p_fisher, min_n)
}

p_fisher <- map_df(
    all_of(nominal_var),
    fisher_map
) %>%
    select(p_fisher = p, min_n = value, variable)

p_norm <- tibble(variable = all_of(nominal_var)) %>%
    left_join(p_chisq, by = "variable") %>%
    left_join(p_fisher, by = "variable") %>%
    group_by(variable) %>%
    mutate(p_final = ifelse(nrow(data_total) > 40 & min_n >= 5, p_chisq, p_fisher))

p_numb
p_norm

p_value <- bind_rows(
    p_numb %>%
        select(variable, p_final),
    p_norm %>%
        select(variable, p_final)
)

# ---------------------------------------- get summary

theme_gtsummary_language("en", big.mark = "")

total_summary <- data_total %>%
    relocate(age, gender, height, weight, CDAI_before, CRP_before, dose) %>%
    mutate(
        fecal_calprotectin = factor(fecal_calprotectin, levels = c(0, 1, 2)),
        Montreal_age = factor(Montreal_age, levels = c(1, 2, 3)),
        Montreal_L = factor(Montreal_L, levels = c(1, 2, 3, 4, 5, 6)),
        Montreal_B = factor(Montreal_B, levels = c(1, 2, 3, 4)),
        Montreal_P = factor(Montreal_P, levels = c(0, 1))
    ) %>%
    tbl_summary(
        by = group,
        statistic = list(
            all_continuous2() ~ c("{mean} \u00B1 {sd}", "{median} ({p25}, {p75})", "{p_miss}"),
            all_categorical() ~ "{n} ({p}%)"
        ),
        digits = list(
            all_continuous() ~ 1,
            all_categorical() ~ c(0, 1)
        ),
        value = list(
            ADA ~ "1",
            Azathioprine ~ "1",
            Mesalazine ~ "1",
            Thalidomide ~ "1",
            MTX ~ "1"
        )
    ) %>%
    as_tibble() %>%
    rename("variable" = `**Characteristic**`) %>%
    left_join(p_value) %>%
    mutate(p_final = ifelse(p_final >= 0.001, round(p_final, 3), "< 0.001")) %>%
    mutate(across(where(is.numeric), as.character)) %>%
    mutate(variable = case_when(
        variable == "age" ~ "Age (year)",
        variable == "gender" ~ "Gender",
        variable == "height" ~ "Height(cm)",
        variable == "weight" ~ "Weight(kg)",
        variable == "CDAI_before" ~ "CDAI",
        variable == "CRP_before" ~ "CRP",
        variable == "dose" ~ "IFX dose(mg)",
        variable == "D_dimer" ~ "D-dimer(μg/L)",
        variable == "Fg" ~ "Fibrinogen(g/L)",
        variable == "fecal_calprotectin" ~ "Fecal calprotectin",
        variable == "Montreal_age" ~ "Montreal Classification(age)",
        variable == "Montreal_L" ~ "Montreal Classification(L)",
        variable == "Montreal_B" ~ "Montreal Classification(B)",
        variable == "Montreal_P" ~ "Montreal Classification(P)",
        variable == "response" ~ "Clinical response",
        .default = variable
    ))

total_summary %>%
    replace(is.na(.), "") %>%
    write_csv(file = "output/split_comparison.csv")
