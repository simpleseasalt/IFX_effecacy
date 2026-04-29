# | ---------------------------------------
# | Author: Simplezzz
# | Date: 2025-08-11 12:58:07
# | LastEditTime: 2026-04-29 21:25:09
# | FilePath: \R_scripts\3_model_construction.R
# | Description: 
# | ---------------------------------------

library(tidyverse)
library(tidymodels)
library(themis)
library(stacks)
library(future)
library(readxl)
library(dcurves)

tidymodels_prefer()

set.seed(2025)

# ---------------------------------------- load data

load("output/1_data_tidy.RData")

load("output/data_model.RData")

# ---------------------------------------- data split

IFX_split <- initial_split(data_tidy, prop = 0.7, strata = group)

IFX_train <- training(IFX_split)

IFX_validation <- testing(IFX_split)

variable_include <- names(data_model)

IFX_recipe <- recipe(group ~ ESR + Montreal_age + CRP_before + RBC, data = IFX_train) %>%
    step_normalize(CDAI_before, ESR, CRP_before, RBC) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_smote(group, over_ratio = 1, seed = 2025)

save(IFX_train, file = "output/IFX_train.RData")
save(IFX_validation, file = "output/IF.RData")

IFX_recipe_prep <- IFX_recipe %>%
    prep()

save(IFX_recipe_prep, file = "output/IFX_recipe.RData")

# ---------------------------------------- model settings

mod_plr <- logistic_reg(penalty = tune(), mixture = tune()) %>%
    set_engine("glmnet") %>%
    set_mode("classification")

mod_svm <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
    set_engine("kernlab") %>%
    set_mode("classification")

mod_rf <- rand_forest(min_n = tune(), trees = tune()) %>%
    set_engine("randomForest") %>%
    set_mode("classification")

mod_xgb <- boost_tree(
    trees = tune(),
    tree_depth = tune(),
    min_n = tune(),
    sample_size = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
) %>%
    set_engine("xgboost") %>%
    set_mode("classification")

## --------------------------------------- metrics

metrics_result <- metric_set(roc_auc, accuracy, precision, sensitivity, specificity, recall, f_meas)

options(yardstick.event_level = "second")

## ---------------------------------------- cross validaion

IFX_cv <- vfold_cv(IFX_train, strata = "group", repeats = 1, v = 10)

## --------------------------------------- define multiprocess

plan(multisession, workers = availableCores() - 2)

## ---------------------------------------- workflow

IFX_wf <- workflow_set(
    preproc = list(
        recipe = IFX_recipe
    ),
    models = list(
        SVM = mod_svm,
        EN = mod_plr,
        RF = mod_rf,
        XGB = mod_xgb
    ),
    cross = FALSE
) %>%
    mutate(
        wflow_id = case_when(
            wflow_id == "recipe_SVM" ~ "SVM",
            wflow_id == "recipe_EN" ~ "EN",
            wflow_id == "recipe_RF" ~ "RF",
            wflow_id == "recipe_XGB" ~ "XGBoost",
        )
    )

tune_control <- control_bayes(
    seed = 2025,
    allow_par = TRUE,
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
)

grid_result <- workflow_map(
    IFX_wf,
    fn = "tune_bayes",
    resamples = IFX_cv,
    metrics = metrics_result,
    verbose = FALSE,
    iter = 25,
    initial = 10,
    seed = 2025,
    control = tune_control
)

autoplot(
    grid_result,
    rank_metric = "roc_auc"
) +
    theme_bw()

updated_mods_plot <- autoplot(
    grid_result,
    metric = "roc_auc",
    rank_metric = "roc_auc"
) +
    labs(x = "Model Rank", y = "AUROC") +
    scale_shape_discrete(guide = "none") +
    scale_color_discrete(name = "Models", label = c("XGBoost", "EN", "RF", "SVM")) +
    theme_bw() +
    theme(
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 18, face = "bold"),
        legend.position = "right"
    )

updated_mods_plot

tiff(filename = "plot/updated_mods_plot.tiff", width = 10, height = 10, res = 300, units = "in", compression = "lzw")

updated_mods_plot

dev.off()

## ---------------------------------------- export result
### ---------------------------------------- glm

EN_tune <- grid_result %>%
    extract_workflow_set_result("EN") %>%
    unnest(cols = .metrics) %>%
    filter(.metric == "roc_auc") %>%
    group_by(penalty) %>%
    summarise(
        mean = mean(.estimate),
        sd = sd(.estimate)
    )

write.csv(EN_tune, "output/EN_tune.csv")

### ---------------------------------------- svm

SVM_tune <- grid_result %>%
    extract_workflow_set_result("SVM") %>%
    unnest(cols = .metrics) %>%
    filter(.metric == "roc_auc") %>%
    group_by(cost) %>%
    summarise(
        mean = mean(.estimate),
        sd = sd(.estimate)
    )

write.csv(SVM_tune, "output/SVM_tune.csv")

### ---------------------------------------- rf

RF_tune <- grid_result %>%
    extract_workflow_set_result("RF") %>%
    unnest(cols = .metrics) %>%
    filter(.metric == "roc_auc") %>%
    group_by(min_n, trees) %>%
    summarise(
        mean = mean(.estimate),
        sd = sd(.estimate)
    )

write.csv(RF_tune, "output/RF_tune.csv")

### ---------------------------------------- xgb

XGB_tune <- grid_result %>%
    extract_workflow_set_result("XGBoost") %>%
    unnest(cols = .metrics) %>%
    filter(.metric == "roc_auc") %>%
    group_by(min_n, tree_depth, learn_rate, loss_reduction) %>%
    summarise(
        mean = mean(.estimate),
        sd = sd(.estimate)
    )

write.csv(XGB_tune, "output/XGB_tune.csv")

# ---------------------------------------- fit on validation data

EN_fit_validation <- finalize_workflow(
    extract_workflow(grid_result, id = "EN"),
    select_best(
        grid_result[grid_result$wflow_id == "EN", "result"][[1]][[1]],
        metric = "roc_auc"
    )
) %>%
    last_fit(
        split = IFX_split,
        metrics = metrics_result
    )

XGB_fit_validation <- finalize_workflow(
    extract_workflow(grid_result, id = "XGBoost"),
    select_best(
        grid_result[grid_result$wflow_id == "XGBoost", "result"][[1]][[1]],
        metric = "roc_auc"
    )
) %>%
    last_fit(
        split = IFX_split,
        metrics = metrics_result
    )

RF_fit_validation <- finalize_workflow(
    extract_workflow(grid_result, id = "RF"),
    select_best(
        grid_result[grid_result$wflow_id == "RF", "result"][[1]][[1]],
        metric = "roc_auc"
    )
) %>%
    last_fit(
        split = IFX_split,
        metrics = metrics_result
    )

SVM_fit_validation <- finalize_workflow(
    extract_workflow(grid_result, id = "SVM"),
    select_best(
        grid_result[grid_result$wflow_id == "SVM", "result"][[1]][[1]],
        metric = "roc_auc"
    )
) %>%
    last_fit(
        split = IFX_split,
        metrics = metrics_result
    )

fit_validation_list <- list(
    EN_fit_validation,
    RF_fit_validation,
    SVM_fit_validation,
    XGB_fit_validation
)

fit_validation_res <- fit_validation_list %>%
    map(collect_metrics) %>%
    rlist::list.stack() %>%
    mutate(model = c(
        rep("EN", nrow(.) / 4),
        rep("RF", nrow(.) / 4),
        rep("SVM", nrow(.) / 4),
        rep("XGBoost", nrow(.) / 4)
    )) %>%
    select(-c(.estimator, .config)) %>%
    pivot_wider(
        names_from = .metric,
        values_from = .estimate
    ) %>%
    mutate(Dataset = "Validation")

# ---------------------------------------- fit on test set

IFX_test <- read_excel("data/IFX-testset.xlsx") %>%
    filter(CDAI_before >= 70) %>%
    select(starts_with("CDAI"), CRP_before, ESR, RBC, Montreal_age = Age) %>%
    mutate(
        group = ifelse(CDAI_after - CDAI_before >= -70, 0, 1),
        group = as.factor(group),
        Montreal_age = factor(Montreal_age, levels = c("2", "1", "3"))
    )

save(IFX_test, file = "output/IFX_test.RData")

new_metrics_list <- list()

for (model_id in c("EN", "SVM", "RF", "XGBoost")) {

    best_params <- select_best(
        grid_result[grid_result$wflow_id == model_id, "result"][[1]][[1]],
        metric = "roc_auc"
    )

    final_wf <- finalize_workflow(extract_workflow(grid_result, id = model_id), best_params)

    fitted_wf <- fit(final_wf, data = IFX_train)

    pred_class <- predict(fitted_wf, new_data = IFX_test, type = "class")
    pred_prob <- predict(fitted_wf, new_data = IFX_test, type = "prob")

    results <- IFX_test %>%
        select(group) %>%
        bind_cols(pred_class, pred_prob) %>%
        mutate(.pred_class = factor(.pred_class, levels = c("0", "1")))

    metrics <- metrics_result(results, truth = group, estimate = .pred_class, .pred_0) %>%
        mutate(model = model_id)

    new_metrics_list[[model_id]] <- metrics
}

fit_test_res <- bind_rows(new_metrics_list) %>%
    select(model, .metric, .estimate) %>%
    pivot_wider(names_from = .metric, values_from = .estimate) %>%
    mutate(Dataset = "Test")

base_models_performance <- bind_rows(fit_validation_res, fit_test_res) %>%
    rename(Model = model) %>%
    arrange(Model, Dataset)

base_models_performance

# ---------------------------------------- stack model

IFX_stack <- stacks() %>%
    add_candidates(extract_workflow_set_result(grid_result, "EN"), name = "EN") %>%
    add_candidates(extract_workflow_set_result(grid_result, "RF"), name = "RF")

IFX_stack_fit <- IFX_stack %>%
    blend_predictions(
        metric = metrics_result,
        control = control_grid(
            allow_par = TRUE
        )
    ) %>%
    fit_members()

stack_validation_pred <- predict(IFX_stack_fit, IFX_validation) %>%
    bind_cols(predict(IFX_stack_fit, IFX_validation, type = "prob")) %>%
    bind_cols(IFX_validation %>% select(group)) %>%
    mutate(wflow_id = "Stack model")

stack_test_pred <- predict(IFX_stack_fit, IFX_test) %>%
    bind_cols(predict(IFX_stack_fit, IFX_test, type = "prob")) %>%
    bind_cols(IFX_test %>% select(group)) %>%
    mutate(wflow_id = "Stack model")

stack_validation_metrics <- metrics_result(stack_validation_pred, truth = group, estimate = .pred_class, .pred_0) %>%
    mutate(
        Dataset = "Validation",
        Model = "Stack model"
    )

stack_test_metrics <- metrics_result(stack_test_pred, truth = group, estimate = .pred_class, .pred_0) %>%
    mutate(
        Dataset = "Test",
        Model = "Stack model"
    )

stackl_all_metrics <- bind_rows(stack_validation_metrics, stack_test_metrics) %>%
    pivot_wider(
        names_from = ".metric",
        values_from = ".estimate"
    ) %>%
    select(-.estimator) %>%
    mutate(across(accuracy:roc_auc, ~ format(., digits = 3)))

stackl_all_metrics

# ---------------------------------------- calculate CI of AUROC

calc_auc_ci <- function(workflow_fit, data, n_boot = 1000) {
    set.seed(2025)

    boots <- bootstraps(data, times = n_boot, strata = group)

    boot_auc <- boots %>%
        mutate(auc_results = map(splits, function(s) {
            d <- analysis(s)
            predict(workflow_fit, d, type = "prob") %>%
                bind_cols(d %>% select(group)) %>%
                roc_auc(truth = group, .pred_0) %>%
                rename(estimate = .estimate) %>%
                mutate(term = "roc_auc") %>%
                select(term, estimate)
        }))

    ci <- int_pctl(boot_auc, auc_results) %>%
        mutate(ci_label = paste0(round(.lower, 3), " - ", round(.upper, 3)))

    return(ci)
}

base_summary_list <- list()

model_ids <- c("EN", "SVM", "RF", "XGBoost")

for (id in model_ids) {
    actual_id <- if (id == "XGBoost") "XGBoost" else id

    best_params <- select_best(
        grid_result[grid_result$wflow_id == actual_id, "result"][[1]][[1]],
        metric = "roc_auc"
    )

    base_wf <- finalize_workflow(extract_workflow(grid_result, id = actual_id), best_params)

    fitted_model <- fit(base_wf, data = IFX_train)

    datasets <- list(Validation = IFX_validation, Test = IFX_test)

    for (ds_name in names(datasets)) {
        target_data <- datasets[[ds_name]]

        preds <- predict(fitted_model, target_data, type = "class") %>%
            bind_cols(predict(fitted_model, target_data, type = "prob")) %>%
            bind_cols(target_data %>% select(group))

        metrics_tab <- metrics_result(preds, truth = group, estimate = .pred_class, .pred_0) %>%
            select(.metric, .estimate) %>%
            pivot_wider(names_from = .metric, values_from = .estimate)

        auc_ci_res <- calc_auc_ci(fitted_model, target_data)

        res_combined <- metrics_tab %>%
            mutate(
                Model = id,
                Dataset = ds_name,
                AUROC_95_CI = auc_ci_res$ci_label
            ) %>%
            select(Model, Dataset, roc_auc, AUROC_95_CI, everything())

        base_summary_list[[paste(id, ds_name)]] <- res_combined
    }
}

base_performance_table <- bind_rows(base_summary_list)

print(base_performance_table)

base_performance_table %>%
    mutate(across(c(roc_auc, accuracy, precision, sensitivity, specificity, recall, f_meas), ~ format(., digits = 3))) %>%
    write.csv("output/Base_Models_Performance_with_CI.csv")

# ---------------------------------------- 

all_models_fit <- list(
    EN = EN_fit_validation$.workflow[[1]],
    SVM = SVM_fit_validation$.workflow[[1]],
    RF = RF_fit_validation$.workflow[[1]],
    XGBoost = XGB_fit_validation$.workflow[[1]],
    Stacking = IFX_stack_fit
)

all_validation_pred <- imap_dfr(
    all_models_fit, ~ 
        mutate(
            predict(., IFX_validation, type = "prob"),
            predict(., IFX_validation),
            Dataset = "Validation",
            Model = .y) %>%
        bind_cols(IFX_validation %>% select(group))
)

all_test_pred <- imap_dfr(
    all_models_fit, ~ 
        mutate(
            predict(., IFX_test, type = "prob"),
            predict(., IFX_test),
            Dataset = "Test",
            Model = .y) %>%
        bind_cols(IFX_test %>% select(group))
)

all_pred <- bind_rows(
    all_validation_pred,
    all_test_pred
) %>%
    mutate(
        Model = fct_relevel(Model, c("EN", "SVM", "RF", "XGBoost", "Stacking")),
        Dataset = fct_relevel(Dataset, c("Validation", "Test"))
    )

auc_ci <- all_pred %>%
    group_by(Model, Dataset) %>%
    summarise(
        roc_obj = list(roc(
            response = group, predictor = .pred_1,
            levels = c("0", "1"), direction = "<"
        )),
        ci_obj = list(ci.auc(roc_obj[[1]], conf.level = 0.95, method = "bootstrap", boot.n = 1000)),
        .groups = "drop"
    ) %>%
    mutate(
        auc = map_dbl(ci_obj, ~ as.numeric(.x)[2]), # AUC 点估计
        auc_lower = map_dbl(ci_obj, ~ as.numeric(.x)[1]),
        auc_upper = map_dbl(ci_obj, ~ as.numeric(.x)[3])
    ) %>%
    select(Model, Dataset, auc_lower, auc_upper)

all_metrics <- bind_rows(
    all_validation_pred,
    all_test_pred
) %>%
    group_by(Model, Dataset) %>%
    metrics_result(truth = group, estimate = .pred_class, .pred_0) %>%
    pivot_wider(
        names_from = .metric,
        values_from = .estimate
    ) %>%
    left_join(auc_ci, by = c("Model", "Dataset")) %>%
    mutate(
        Model = factor(Model, levels = c("EN", "SVM", "RF", "XGBoost", "Stacking")),
        Dataset = factor(Dataset, levels = c("Validation", "Test"))
    ) %>%
    arrange(Model, Dataset)

all_metrics

all_metrics %>%
    select(-.estimator) %>%
    mutate(
        across(where(is.numeric), ~ format(., digits = 3)),
        `AUROC (95% CI)` = paste0(auc_lower, " - ", auc_upper)
    ) %>%
    write_csv("output/All_Models_Performance_with_CI.csv")

# --------------------------------------- plot

theme_journal <- function() {
    theme_bw() +
        theme(
            axis.title = element_text(size = 18, face = "bold"),
            axis.text = element_text(size = 14),
            legend.text = element_text(size = 14),
            legend.title = element_text(size = 18, face = "bold"),
            legend.position = "right",
            strip.background = element_rect(fill = "gray95"),
            strip.text = element_text(size = 14, face = "bold"),
            panel.grid.minor = element_blank()
        )
}

## --------------------------------------- ARROC
### --------------------------------------- validation set

validation_labels <- all_metrics %>%
    filter(Dataset == "Validation") %>%
    select(Model, roc_auc) %>%
    mutate(
        label = paste(Model, format(roc_auc, digits = 3), sep = " ")
    ) %>%
    pull(label) %>%
    paste(collapse = "\n") %>%
    paste("AUROC\n", ., sep = "", collapse = "\n") %>%
    as_tibble()

plot_roc_validation <- all_pred %>%
    filter(Dataset == "Validation") %>%
    group_by(Model) %>%
    roc_curve(truth = group, .pred_0) %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity, color = Model)) +
    geom_path(size = 1.1) +
    geom_abline(lty = 3, color = "gray50") +
    geom_text(
        data = validation_labels,
        aes(x = 0.6, y = 0.25, label = value),
        hjust = 0,
        size = 6,
        inherit.aes = FALSE
    ) +
    labs(
        x = "1 - Specificity",
        y = "Sensitivity",
        color = "Models"
    ) +
    theme_bw() +
    theme_journal()

plot_roc_validation

tiff(filename = "plot/plot_roc_validation.tiff", width = 8, height = 8, res = 300, units = "in", compression = "lzw")

plot_roc_validation

dev.off()

### --------------------------------------- test set

test_labels <- all_metrics %>%
    filter(Dataset == "Test") %>%
    select(Model, roc_auc) %>%
    mutate(
        label = paste(Model, format(roc_auc, digits = 3), sep = " ")
    ) %>%
    pull(label) %>%
    paste(collapse = "\n") %>%
    paste("AUROC\n", ., collapse = "\n") %>%
    as_tibble()

plot_roc_test <- all_pred %>%
    filter(Dataset == "Test") %>%
    group_by(Model) %>%
    roc_curve(truth = group, .pred_0) %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity, color = Model)) +
    geom_path(size = 1.1) +
    geom_abline(lty = 3, color = "gray50") +
    geom_text(
        data = test_labels,
        aes(x = 0.6, y = 0.25, label = value),
        hjust = 0,
        size = 6,
        inherit.aes = FALSE
    ) +
    labs(
        x = "1 - Specificity",
        y = "Sensitivity",
        color = "Models"
    ) +
    theme_bw() +
    theme_journal()

plot_roc_test

tiff(filename = "plot/plot_roc_test.tiff", width = 8, height = 8, res = 300, units = "in", compression = "lzw")

plot_roc_test

dev.off()

## ---------------------------------------- plot calibration curve

library(probably)

plot_cal_validation <- all_pred %>%
    filter(Dataset == "Validation") %>%
    cal_plot_breaks(truth = group, estimate = .pred_0, num_breaks = 8, .by = Model) +
    theme_journal() +
    theme(
        panel.spacing = unit(1.5, "lines")
    )

plot_cal_validation

tiff(filename = "plot/plot_calbration_curve.tiff", width = 8, height = 8, res = 300, units = "in", compression = "lzw")

plot_cal_validation

dev.off()

# ---------------------------------------- plot clincal decision curve

library(dcurves)

dca_data_prep <- all_pred %>%
    group_by(Model, Dataset) %>%
    mutate(row_id = row_number()) %>%
    ungroup() %>%
    select(row_id, Model, Dataset, group, .pred_1) %>%
    pivot_wider(names_from = Model, values_from = .pred_1) %>%
    mutate(group = as.numeric(as.character(group))) %>%
    select(-row_id)

plot_dca_validation <- dca_data_prep %>%
    filter(Dataset == "Validation") %>%
    dca(
        data = .,
        group ~ EN + SVM + RF + XGBoost + Stacking,
        label = list(
            EN = "Elastic Net",
            SVM = "SVM",
            RF = "Random Forest",
            XGBoost = "XGBoost",
            Stacking = "Stack Model"
        )
    ) %>%
    plot(smooth = TRUE) +
    theme_journal() +
    labs(
        x = "Threshold Probability",
        y = "Net Benefit"
    )

plot_dca_validation

tiff(filename = "plot/plot_dca_validation.tiff", width = 8, height = 8, res = 300, units = "in", compression = "lzw")

plot_dca_validation

dev.off()

plot_dca_test <- dca_data_prep %>%
    filter(Dataset == "Test") %>%
    dca(
        data = .,
        group ~ EN + SVM + RF + XGBoost + Stacking,
        label = list(
            EN = "Elastic Net",
            SVM = "SVM",
            RF = "Random Forest",
            XGBoost = "XGBoost",
            Stacking = "Stack Model"
        )
    ) %>%
    plot(smooth = TRUE) +
    theme_journal() +
    labs(
        x = "Threshold Probability",
        y = "Net Benefit"
    )

plot_dca_test

tiff(filename = "plot/plot_dca_test.tiff", width = 8, height = 8, res = 300, units = "in", compression = "lzw")

plot_dca_test

dev.off()

## --------------------------------------- PR curve

pr_auc_metrics <- all_pred %>%
    group_by(Model, Dataset) %>%
    pr_auc(truth = group, .pred_0) %>%
    mutate(
        label = paste(Model, format(.estimate, digits = 3), sep = " ")
    )

pr_curve_data <- all_pred %>%
    group_by(Model, Dataset) %>%
    pr_curve(truth = group, .pred_0)

### --------------------------------------- validation set

label_pr_validation <- pr_auc_metrics %>%
    filter(Dataset == "Validation") %>%
    arrange(Model) %>%
    mutate(text_line = paste0(Model, sep = " ", format(.estimate, digits = 3))) %>%
    pull(text_line) %>%
    paste(collapse = "\n") %>%
    paste("AUPRC\n", ., sep = "") %>%
    as_tibble()

plot_pr_validation <- pr_curve_data %>%
    filter(Dataset == "Validation") %>%
    ggplot(aes(x = recall, y = precision, color = Model)) +
    geom_path(size = 1.1) +
    geom_text(
        data = label_pr_validation,
        aes(x = 0.1, y = 0.25, label = value),
        hjust = 0,
        size = 6,
        color = "black",
        inherit.aes = FALSE
    ) +
    xlim(0, 1) +
    ylim(0, 1) +
    labs(
        x = "Recall",
        y = "Precision",
        color = "Models"
    ) +
    theme_journal()

tiff(filename = "plot/plot_pr_validation.tiff", width = 8, height = 8, res = 300, units = "in", compression = "lzw")

plot_pr_validation

dev.off()

### --------------------------------------- test set

label_pr_test <- pr_auc_metrics %>%
    filter(Dataset == "Test") %>%
    arrange(Model) %>%
    mutate(text_line = paste0(Model, sep = " ", format(.estimate, digits = 3))) %>%
    pull(text_line) %>%
    paste(collapse = "\n") %>%
    paste("AUPRC\n", ., sep = "") %>%
    as_tibble()

plot_pr_test <- pr_curve_data %>%
    filter(Dataset == "Test") %>%
    ggplot(aes(x = recall, y = precision, color = Model)) +
    geom_path(size = 1.1) +
    geom_text(
        data = label_pr_test,
        aes(x = 0.1, y = 0.25, label = value),
        hjust = 0,
        size = 6,
        color = "black",
        inherit.aes = FALSE
    ) +
    xlim(0, 1) +
    ylim(0, 1) +
    labs(
        x = "Recall",
        y = "Precision",
        color = "Models"
    ) +
    theme_journal()

tiff(filename = "plot/plot_pr_test.tiff", width = 8, height = 8, res = 300, units = "in", compression = "lzw")

plot_pr_test

dev.off()

## --------------------------------------- save plot

p1 <- plot_roc_validation + theme(legend.position = "none")
p2 <- plot_roc_test + theme(legend.position = "none")
p3 <- plot_pr_validation + theme(legend.position = "none")
p4 <- plot_pr_test + theme(legend.position = "none")
p5 <- plot_cal_validation + theme(legend.position = "none")
p6 <- plot_dca_validation + theme(legend.position = c(0.2, 0.4))

combined_plot <- (p1 | p2) / (p3 | p4) / (p5 | p6) +
    plot_annotation(
        tag_levels = "A"
    ) &
    theme(
        plot.title = element_text(size = 28, face = "bold"),
        plot.tag = element_text(size = 18, face = "bold")
    )

combined_plot_legend <- plot_grid(
    combined_plot,
    shared_legend,
    ncol = 1,
    labels = NULL,
    rel_heights = c(10, 1)
)

tiff("plot/combined_plot_legend.tiff", width = 15, height = 20, res = 300, units = "in", compression = "lzw")

print(combined_plot_legend)

dev.off()

# ---------------------------------------- Delong's test

base_models <- c("EN", "SVM", "RF", "XGBoost")

delong_results <- list()

for (ds in c("Validation", "Test")) {
    stack_pred <- all_pred %>%
        filter(Model == "Stacking", Dataset == ds) %>%
        pull(.pred_1)
    stack_truth <- all_pred %>%
        filter(Model == "Stacking", Dataset == ds) %>%
        pull(group) %>%
        as.numeric() - 1 

    for (bm in base_models) {
        base_pred <- all_pred %>%
            filter(Model == bm, Dataset == ds) %>%
            pull(.pred_1)
        base_truth <- all_pred %>%
            filter(Model == bm, Dataset == ds) %>%
            pull(group) %>%
            as.numeric() - 1

        if (!identical(stack_truth, base_truth)) {
            warning(paste("真实标签顺序不一致:", ds, bm))
            next
        }

        roc_test <- roc.test(
            roc(stack_truth, stack_pred, quiet = TRUE),
            roc(base_truth, base_pred, quiet = TRUE),
            method = "delong"
        )

        delong_results[[paste(ds, bm, sep = "_")]] <- data.frame(
            Dataset = ds,
            Model_vs = paste("Stacking vs", bm),
            AUC_stack = as.numeric(roc_test$roc1$auc),
            AUC_other = as.numeric(roc_test$roc2$auc),
            Difference = roc_test$estimate[1] - roc_test$estimate[2],
            CI_lower = roc_test$conf.int[1],
            CI_upper = roc_test$conf.int[2],
            p_value = roc_test$p.value
        )
    }
}

delong_summary <- do.call(rbind, delong_results)
rownames(delong_summary)

print(delong_summary)

# ---------------------------------------- save final model
 
final_mod <- IFX_stack_fit

save(final_mod, file = "output/final_mod.Rdata")

predict(final_mod, new_data = IFX_validation[1, ], type = "prob")

# ---------------------------------------- model fairness

stack_validation_predictions <- IFX_validation %>%
    mutate(
        predict(final_mod, new_data = ., type = "prob"),
        predict(final_mod, new_data = ., type = "class"),
        .pred_class = factor(.pred_class, levels = c("0", "1")),
        Montreal_age = factor(Montreal_age, levels = c("1", "2", "3")),
        CDAI_group = case_when(
            CDAI_before < 150 ~ "Remission",
            CDAI_before >= 150 & CDAI_before < 220 ~ "Mild",
            CDAI_before >= 220 & CDAI_before < 450 ~ "Moderate",
            TRUE ~ "Severe"
        ),
        CDAI_group = factor(CDAI_group, levels = c("Remission", "Mild", "Moderate", "Severe"))
    )

## --------------------------------------- age

metrics_by_age <- stack_validation_predictions %>%
    group_by(Montreal_age) %>%
    metrics_result(truth = group, estimate = .pred_class, .pred_0) %>%
    pivot_wider(
        names_from = ".metric",
        values_from = ".estimate"
    ) %>%
    rename("group" = Montreal_age)

## --------------------------------------- gender

metrics_by_gender <- stack_validation_predictions %>%
    group_by(gender) %>%
    metrics_result(truth = group, estimate = .pred_class, .pred_0) %>%
    pivot_wider(
        names_from = ".metric",
        values_from = ".estimate"
    ) %>%
    rename("group" = gender) %>%
    mutate(
        group = case_when(
            group == 1 ~ "Male",
            .default = "Female"
        )
    )

## --------------------------------------- CDAI

metrics_by_CDAI <- stack_validation_predictions %>%
    group_by(CDAI_group) %>%
    metrics_result(truth = group, estimate = .pred_class, .pred_0) %>%
    pivot_wider(
        names_from = ".metric",
        values_from = ".estimate"
    ) %>%
    rename("group" = CDAI_group)

## --------------------------------------- combine and out put

subgroup_number <- bind_cols(
        `Total number` = bind_rows(
            count(stack_validation_predictions, Montreal_age) %>% select(n),
            count(stack_validation_predictions, gender) %>% select(n),
            count(stack_validation_predictions, CDAI_group) %>% select(n)
        ),
        `Non-response` = bind_rows(
            count(stack_validation_predictions, Montreal_age, group) %>% filter(group == 0) %>% select(n),
            count(stack_validation_predictions, gender, group) %>% filter(group == 0) %>% select(n),
            count(stack_validation_predictions, CDAI_group, group) %>% filter(group == 0) %>% select(n),
        )
    )

names(subgroup_number) <- c("Total number", "Clinical response")

subgroup_result <- bind_rows(metrics_by_age, metrics_by_gender, metrics_by_CDAI) %>%
    mutate(across(where(is.numeric), format, digits = 3)) %>%
    select(-.estimator) %>%
    bind_cols(subgroup_number) %>%
    mutate(
        group = case_when(
            group == 1 ~ "Age ≤ 16",
            group == 2 ~ "Age (16 - 40)",
            group == 3 ~ "Age ≥ 40",
            .default = group
        )
    ) %>%
    relocate(c(`Total number`, `Clinical response`), .after = group) %>%
    rename(
        "Accuracy" = "accuracy",
        "Precision" = "precision",
        "Sensitivity" = "sensitivity",
        "Specificity" = "specificity",
        "Recall" = "recall",
        "F score" = "f_meas",
        "AUROC" = "roc_auc"
    )

write_csv(subgroup_result, file = "output/subgroup_result.csv")

# --------------------------------------- 

save.image("output/IFX_workspace.RData")

# ! end

all_model_list
