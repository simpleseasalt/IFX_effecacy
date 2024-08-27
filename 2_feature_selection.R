# | ---------------------------------------
# | Author: Simplezzz
# | Date: 2024-08-04 15:20:38
# | LastEditTime: 2024-08-27 21:53:22
# | FilePath: \script\2_feature_selection.R
# | Description:
# | ---------------------------------------

library(tidyverse)
library(gtsummary)
library(rstatix)
library(nortest)
library(glmnet)
library(ggfortify)

# 1---------------------------------------- load data

load("output/1_data_imputed.RData")

data_imputed <- data_imputed %>%
    select(-c(CDAI_change, monitoring_c)) %>%
    rename(CDAI = CDAI_before)

# 1---------------------------------------- descriptive statistics
# 2---------------------------------------- numeric variables

numeric_var <- data_imputed %>%
    select_if(is.numeric) %>%
    names()

# 3---------------------------------------- normality test

ad.test.multi <- function(x) {
    ad.test(x) %>%
        broom::tidy()
}

normality <- map_df(
    data_imputed[numeric_var],
    ad.test.multi
) %>%
    bind_cols(variable = numeric_var) %>%
    select(-method) %>%
    rename("p_norm" = "p.value")

# 3---------------------------------------- variance test

variance <- map_df(
    data_imputed[numeric_var],
    function(x) {
        levene_test(data_imputed, x ~ data_imputed$validaty)
    }
) %>%
    select(p_vari = p) %>%
    cbind(variable = numeric_var)

# 3---------------------------------------- t test

t_test_res <- data_imputed %>%
    select(all_of(numeric_var), validaty) %>%
    pivot_longer(
        cols = all_of(numeric_var),
        names_to = "variable"
    ) %>%
    group_by(variable) %>%
    t_test(value ~ validaty, var.equal = TRUE) %>%
    select(variable, p_t_test = p)

# 3---------------------------------------- wilcox test

wilcox_test_res <- data_imputed %>%
    select(all_of(numeric_var), validaty) %>%
    pivot_longer(
        cols = all_of(numeric_var),
        names_to = "variable"
    ) %>%
    group_by(variable) %>%
    wilcox_test(value ~ validaty) %>%
    select(variable, p_wilcox_test = p)

p_numb <- normality %>%
    left_join(variance, by = "variable") %>%
    left_join(t_test_res, by = "variable") %>%
    left_join(wilcox_test_res, by = "variable") %>%
    group_by(variable) %>%
    mutate(p_final = if_else(p_norm >= 0.05 & p_vari >= 0.05, p_t_test, p_wilcox_test)) %>%
    add_significance(p.col = "p_final")

# 2---------------------------------------- nominal variables

nominal_var <- data_imputed %>%
    select_if(is.factor) %>%
    select(-validaty) %>%
    names()

p_chisq <- data_imputed %>%
    select(all_of(nominal_var), validaty) %>%
    pivot_longer(
        cols = -validaty,
        names_to = "variable"
    ) %>%
    group_by(variable) %>%
    do(chisq_test(.$validaty, .$value)) %>%
    select(variable, p_chisq = p)

# 2---------------------------------------- fisher

freq <- freq_table(data_imputed, validaty, all_of(nominal_var))

fisher_map <- function(x) {
    temp <- freq_table(data_imputed, validaty, x) %>%
        select(-prop) %>%
        pivot_wider(
            names_from = "validaty",
            values_from = "n"
        ) %>%
        replace(is.na(.), 0)
    p_fisher <- temp %>%
        select(all_of(unique(data_imputed$validaty))) %>%
        fisher_test(simulate.p.value = TRUE)
    min_n <- temp %>%
        pivot_longer(
            cols = all_of(unique(data_imputed$validaty))
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
    mutate(p_final = ifelse(nrow(data_imputed) > 40 & min_n >= 5, p_chisq, p_fisher))

p_numb
p_norm

p_value <- bind_rows(
    p_numb %>%
        select(variable, p_final),
    p_norm %>%
        select(variable, p_final)
)

# 1---------------------------------------- get summary

theme_gtsummary_language("en", big.mark = "")

total_summary <- data_imputed %>%
    tbl_summary(
        by = validaty,
        statistic = list(
            all_continuous2() ~ c("{mean} \u00B1 {sd}", "{median} ({p25}, {p75})", "{p_miss}"),
            all_categorical() ~ "{n} ({p}%)"
        ),
        digits = list(
            all_continuous() ~ 1,
            all_categorical() ~ c(0, 1)
        )
    ) %>%
    as_tibble() %>%
    rename("variable" = `**Characteristic**`) %>%
    left_join(p_value) %>%
    mutate(p_final = ifelse(p_final >= 0.001, round(p_final, 3), "< 0.001")) %>%
    mutate(across(where(is.numeric), as.character))

total_summary %>%
    replace(is.na(.), "") %>%
    write_csv(file = "output/2_total_summary.csv")

# 1---------------------------------------- lasso regression
# 2---------------------------------------- select best lambda

var_include_1 <- p_value %>%
    filter(p_final <= 0.1) %>%
    pull(variable)

data_lasso <- data_imputed %>%
    select(all_of(var_include_1), validaty)

lasso_var <- data_lasso %>%
    select(-validaty) %>%
    as.matrix()

lasso_pred <- data_lasso %>%
    select(validaty) %>%
    mutate(validaty = as.numeric(validaty)) %>%
    as.matrix()

lasso_cv <- cv.glmnet(lasso_var, lasso_pred, alpha = 1, nfold = 20, nlambda = 300, family = "binomial", type.measure = "auc")

lasso_cv_plot <- autoplot(
    lasso_cv,
    label = F,
    color = "steelblue",
    alpha = 0.5
) +
    theme_bw() +
    xlab("Log Lambda") +
    ylab("AUROC") +
    theme(
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        legend.position = "none"
    )

tiff(filename = "plot/lasso_cv_plot.tiff", width = 15, height = 10, res = 300, units = "in", compression = "lzw")

lasso_cv_plot

dev.off()

# 2---------------------------------------- lasso regression with best lambda

best_lambda <- lasso_cv$lambda.min

best_lambda

lasso_fit <- glmnet(x = lasso_var, y = lasso_pred, alpha = 1, lambda = 0.0176, fimaly = "binomial", type.measure = "auc")

coef(lasso_fit) %>%
    round(5)

# 2---------------------------------------- trace plot

library(ggrepel)

trace_mod <- glmnet(x = lasso_var, y = lasso_pred, alpha = 1, fimaly = "binomial", type.measure = "auc")

labels <- coef(lasso_fit) %>%
    cbind(trace_mod$beta[, dim(trace_mod$beta)[2]]) %>%
    as.matrix() %>%
    as.data.frame() %>%
    set_names(c("s0", "y")) %>%
    slice(-1) %>%
    arrange(desc(y)) %>%
    mutate(
        label = c("WBC", "ALB", "Weight", "CREA", "CDAI", "Gender", "RBC", "Age")
    ) %>%
    mutate(
        x = rep(-7.48, dim(trace_mod$beta)[1]),
        x_point = rep(-7.48, dim(trace_mod$beta)[1])
    )

trace_plot <- autoplot(trace_mod, label = F, size = 0.8, xvar = "lambda") +
    geom_hline(aes(yintercept = 0), linetype = 2, linewidth = 0.5, color = "gray") +
    geom_vline(aes(xintercept = log(0.0176)), linetype = 2, linewidth = 0.7, color = "black") +
    scale_x_continuous(limits = c(-9, -1.5)) +
    geom_point(
        data = labels,
        aes(x = x_point, y = y)
    ) +
    geom_text_repel(
        data = labels,
        aes(label = label, x = x, y = y),
        color = "black",
        max.overlaps = 50,
        nudge_x = -0.5,
        direction = "y",
        force = 2,
        hjust = 1,
        size = 6
    ) +
    theme_bw() +
    theme(
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        legend.position = "none"
    )

tiff(filename = "plot/trace_plot.tiff", width = 15, height = 10, res = 300, units = "in", compression = "lzw")

trace_plot

dev.off()

# 2----------------------------------------

library(cowplot)

lasso_grid_plot <- plot_grid(lasso_cv_plot, trace_plot, ncol = 2, align = "hv", labels = "AUTO", label_size = 20)

tiff(filename = "plot/lasso_grid_plot.tiff", width = 15, height = 7, res = 300, units = "in", compression = "lzw")

lasso_grid_plot

dev.off()

# 1---------------------------------------- ouput final data

final_var <- labels %>%
    filter(s0 != 0) %>%
    pull(rowname)

final_data <- data_imputed %>%
    select(all_of(final_var), validaty)

save(final_data, file = "output/final_data.RData")

# ! end