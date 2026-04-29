# | ---------------------------------------
# | Author: Simplezzz
# | Date: 2025-08-11 12:58:07
# | LastEditTime: 2026-04-27 19:05:53
# | FilePath: \R_scripts\3.1_sensitive_analysis.R
# | Description: 
# | ---------------------------------------

library(tidyverse)
library(tidymodels)
library(themis)
library(stacks)
library(future)
library(readxl)

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
    step_normalize(ESR, CRP_before, RBC) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_smote(group, over_ratio = 1, seed = 2025)

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
    mutate(group = "Validation")

# ---------------------------------------- fit on test set

IFX_test <- read_excel("data/IFX-testset.xlsx") %>%
    filter(CDAI_before >= 70) %>%
    select(starts_with("CDAI"), CRP_before, ESR, RBC, Montreal_age = Age) %>%
    mutate(
        group = ifelse(CDAI_after - CDAI_before >= -70, 0, 1),
        group = as.factor(group),
        Montreal_age = factor(Montreal_age, levels = c("2", "1", "3"))
    )

write.csv(IFX_test, "output/IFX_test.csv")

new_metrics_list <- list()

for (model_id in c("EN", "SVM", "RF", "XGBoost")) {
    # 提取该模型的最优参数
    best_params <- select_best(
        grid_result[grid_result$wflow_id == model_id, "result"][[1]][[1]],
        metric = "roc_auc"
    )
    # 提取原始工作流，固化参数
    final_wf <- finalize_workflow(extract_workflow(grid_result, id = model_id), best_params)
    # 用整个训练集重新拟合（也可用 IFX_train）
    fitted_wf <- fit(final_wf, data = IFX_train)

    # 对新数据进行预测（得到类别和概率）
    pred_class <- predict(fitted_wf, new_data = IFX_test, type = "class")
    pred_prob <- predict(fitted_wf, new_data = IFX_test, type = "prob")

    # 合并真实标签
    results <- IFX_test %>%
        select(group) %>%
        bind_cols(pred_class, pred_prob) %>%
        mutate(.pred_class = factor(.pred_class, levels = c("0", "1")))

    # 计算各项指标
    metrics <- metrics_result(results, truth = group, estimate = .pred_class, .pred_0) %>%
        mutate(model = model_id)

    new_metrics_list[[model_id]] <- metrics
}

# 合并所有模型的指标，并转换为宽表
fit_test_res <- bind_rows(new_metrics_list) %>%
    select(model, .metric, .estimate) %>%
    pivot_wider(names_from = .metric, values_from = .estimate) %>%
    mutate(group = "Test")

# ---------------------------------------- calculate CI of AUROC

calc_auc_ci <- function(workflow_fit, data, n_boot = 1000) {
    set.seed(2025)

    # 创建 bootstrap 样本 [cite: 1]
    boots <- bootstraps(data, times = n_boot, strata = group)

    # 对每个样本计算 AUC，并转换为 int_pctl 要求的格式
    boot_auc <- boots %>%
        mutate(auc_results = map(splits, function(s) {
            d <- analysis(s)
            predict(workflow_fit, d, type = "prob") %>%
                bind_cols(d %>% select(group)) %>%
                roc_auc(truth = group, .pred_0) %>%
                # 核心修复：重命名列并添加 term 列
                rename(estimate = .estimate) %>%
                mutate(term = "roc_auc") %>%
                select(term, estimate)
        }))

    # 现在可以正确运行 int_pctl
    ci <- int_pctl(boot_auc, auc_results) %>%
        mutate(ci_label = paste0(round(.lower, 3), " - ", round(.upper, 3)))

    return(ci)
}

final_summary_list <- list()
model_ids <- c("EN", "SVM", "RF", "XGBoost")

for (id in model_ids) {
    actual_id <- if (id == "XGBoost") "XGBoost" else id

    best_params <- select_best(
        grid_result[grid_result$wflow_id == actual_id, "result"][[1]][[1]],
        metric = "roc_auc"
    )

    final_wf <- finalize_workflow(extract_workflow(grid_result, id = actual_id), best_params)

    fitted_model <- fit(final_wf, data = IFX_train)

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

        final_summary_list[[paste(id, ds_name)]] <- res_combined
    }
}

final_performance_table <- bind_rows(final_summary_list)

print(final_performance_table)

write.csv(final_performance_table, "output/sensitive_analysis_result.csv", row.names = FALSE)

# ---------------------------------------- plot AUROC

all_roc_combined <- list()
auc_labels_combined <- list()

model_ids <- c("EN", "SVM", "RF", "XGBoost")

for (id in model_ids) {
    best_params <- select_best(
        grid_result[grid_result$wflow_id == id, "result"][[1]][[1]],
        metric = "roc_auc"
    )
    final_wf <- finalize_workflow(extract_workflow(grid_result, id = id), best_params)
    fitted_model <- fit(final_wf, data = IFX_train)

    eval_sets <- list(Validation = IFX_validation, Test = IFX_test)

    for (set_name in names(eval_sets)) {
        preds <- predict(fitted_model, eval_sets[[set_name]], type = "prob") %>%
            bind_cols(eval_sets[[set_name]] %>% select(group))

        roc_df <- roc_curve(preds, truth = group, .pred_0) %>%
            mutate(model = id, dataset = set_name)

        auc_val <- roc_auc(preds, truth = group, .pred_0)$.estimate
        auc_df <- tibble(
            model = id,
            dataset = set_name,
            auc_text = paste0(set_name, " ", round(auc_val, 3))
        )

        all_roc_combined[[paste(id, set_name)]] <- roc_df
        auc_labels_combined[[paste(id, set_name)]] <- auc_df
    }
}

plot_roc_data <- bind_rows(all_roc_combined)
plot_auc_labels <- bind_rows(auc_labels_combined) %>%
    group_by(model) %>%
    summarise(label = paste(auc_text, collapse = "\n"))

final_comparison_plot <- ggplot(plot_roc_data, aes(x = 1 - specificity, y = sensitivity, color = dataset)) +
    geom_abline(lty = 3, color = "gray50") +
    geom_path(size = 1.1) +
    facet_wrap(~model) +
    geom_text(
        data = plot_auc_labels,
        aes(x = 0.65, y = 0.15, label = label),
        inherit.aes = FALSE, color = "black", size = 4.5, hjust = 0
    ) +
    scale_color_manual(values = c("Validation" = "#00AFBB", "Test" = "#FC4E07")) +
    labs(
        x = "1 - Specificity",
        y = "Sensitivity",
        color = "Dataset"
    ) +
    theme_bw() +
    theme(
        legend.position = "bottom",
        strip.background = element_rect(fill = "gray95"),
        strip.text = element_text(size = 14, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        axis.text = element_text(size = 14),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 18, face = "bold"),
        panel.grid.minor = element_blank()
    )

print(final_comparison_plot)

ggsave("plot/ROC_Validation_vs_Test.tiff", width = 10, height = 10, dpi = 300, compression = "lzw")

# ---------------------------------------- stack model

stack_wf <- workflow_set(
    preproc = list(
        recipe = IFX_recipe
    ),
    models = list(
        RF = mod_rf,
        SVM = mod_svm,
        EN = mod_plr
    ),
    cross = FALSE
) %>%
    mutate(
        wflow_id = case_when(
            wflow_id == "recipe_RF" ~ "RF",
            wflow_id == "recipe_SVM" ~ "SVM",
            wflow_id == "recipe_EN" ~ "EN"
        )
    )

stack_result <- workflow_map(
    stack_wf,
    fn = "tune_bayes",
    resamples = IFX_cv,
    metrics = metrics_result,
    verbose = FALSE,
    iter = 25,
    initial = 15,
    control = control_bayes(
        seed = 2025,
        no_improve = 10L,
        allow_par = TRUE,
        save_pred = TRUE,
        parallel_over = "everything",
        save_workflow= TRUE
    )
)

IFX_stack <- stacks() %>%
    add_candidates(extract_workflow_set_result(stack_result, "RF"), name = "RF") %>%
    add_candidates(extract_workflow_set_result(stack_result, "SVM"), name = "SVM") %>%
    add_candidates(extract_workflow_set_result(stack_result, "EN"), name = "EN")

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

validation_metrics <- metrics_result(stack_validation_pred, truth = group, estimate = .pred_class, .pred_0) %>%
    mutate(
        data_type = "Validation",
        wflow_id = "Stack model"
    )

test_metrics <- metrics_result(stack_test_pred, truth = group, estimate = .pred_class, .pred_0) %>%
    mutate(
        data_type = "Test",
        wflow_id = "Stack model"
    )

all_metrics <- bind_rows(validation_metrics, test_metrics) %>%
    pivot_wider(
        names_from = ".metric",
        values_from = ".estimate"
    ) %>%
    select(-.estimator) %>%
    mutate(across(accuracy:roc_auc, ~ format(., digits = 3)))

all_metrics

write.csv(all_metrics, file = "output/result_stack_model.csv")

# --------------------------------------- output all result
## --------------------------------------- metrics

metrics_train_all <- bind_rows(
    updated_result %>%
        select(wflow_id, .metric, mean) %>%
        group_by(wflow_id) %>%
        pivot_wider(
            names_from = ".metric",
            values_from = "mean"
        ),
    train_metrics %>%
        select(wflow_id, .metric, mean = .estimate) %>%
        pivot_wider(
            names_from = ".metric",
            values_from = "mean"
        )
) %>%
    mutate(Dataset = "Train")

metrics_validation_all <- bind_rows(
    last_fit_res %>%
        rename("wflow_id" = "model"),
    validation_metrics %>%
        select(wflow_id, .metric, mean = .estimate) %>%
        pivot_wider(
            names_from = ".metric",
            values_from = "mean"
        )
) %>%
    mutate(Dataset = "Test")

final_result <- bind_rows(
    metrics_train_all,
    metrics_validation_all
) %>%
    relocate(
        wflow_id,
        Dataset
    ) %>%
    relocate(
        f_meas, roc_auc,
        .after = last_col()
    ) %>%
    rename(
        "Model" = "wflow_id",
        "Accuracy" = "accuracy",
        "Precision" = "precision",
        "Sensitivity" = "sensitivity",
        "Specificity" = "specificity",
        "Recall" = "recall",
        "F score" = "f_meas",
        "AUROC" = "roc_auc"
    ) %>%
    arrange(Model) %>%
    mutate(across(c(Accuracy:AUROC), format, nsmall = 3, digits = 3))

final_result %>%
    write_csv("output/final_result.csv")




## --------------------------------------- roc plot
### -------------------------------------- trainset

roc_labels <- final_result %>%
    select(wflow_id, Dataset, roc_auc) %>%
    pivot_wider(
        names_from = Dataset,
        values_from = roc_auc
    ) %>%
    mutate(across(c(Train, Test), format, nsmall = 3, digits = 3)) %>%
    mutate(
        label_train = paste(
            wflow_id,
            Train
        ),
        label_validation = paste(
            wflow_id,
            Test
        )
    )

label_train <- str_c(
    "AUROC\n",
    str_c(roc_labels$label_train, collapse = "\n"),
    collapse = ""
    ) %>%
    as_tibble()

plot_roc_train <- prediction_in_best_mods %>%
    select(wflow_id, names(stack_train_pred)) %>%
    bind_rows(stack_train_pred) %>%
    group_by(wflow_id) %>%
    roc_curve(
        group,
        .pred_0
    ) %>%
    left_join(roc_labels) %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity, color = wflow_id)) +
    geom_path(lwd = 1) +
    geom_abline(lty = 3) +
    geom_text(
        data = label_train,
        aes(x = 0.6, y = 0.25, label = value),
        color = "black",
        size = 6,
        hjust = 0,
        check_overlap = T
    ) +
    coord_equal() +
    theme_bw() +
    scale_color_discrete(name = "Models") +
    labs(x = "1 - Specificity", y = "Sensitivity") +
    theme(
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 18, face = "bold"),
        legend.position = "right"
    )

plot_roc_train

tiff(filename = "plot/plot_roc_train.tiff", width = 10, height = 10, res = 300, units = "in", compression = "lzw")

plot_roc_train

dev.off()

## --------------------------------------- validationset

label_validation <- str_c(
    "AUROC\n",
    str_c(roc_labels$label_validation, collapse = "\n"),
    collapse = ""
) %>%
    as_tibble()

plot_roc_validation <- prediction_in_best_mods %>%
    select(wflow_id, names(stack_validation_pred)) %>%
    bind_rows(stack_validation_pred) %>%
    group_by(wflow_id) %>%
    roc_curve(
        group,
        .pred_0
    ) %>%
    left_join(roc_labels) %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity, color = wflow_id)) +
    geom_path(lwd = 1) +
    geom_abline(lty = 3) +
    geom_text(
        data = label_validation,
        aes(x = 0.6, y = 0.25, label = value),
        color = "black",
        size = 6,
        hjust = 0,
        check_overlap = T
    ) +
    coord_equal() +
    theme_bw() +
    scale_color_discrete(name = "Models") +
    labs(x = "1 - Specificity", y = "Sensitivity") +
    theme(
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 18, face = "bold"),
        legend.position = "right"
    )

plot_roc_validation

tiff(filename = "plot/plot_roc_validation.tiff", width = 10, height = 10, res = 300, units = "in", compression = "lzw")

plot_roc_validation

dev.off()

# ---------------------------------------- save stack model

final_mod <- IFX_stack_fit

save(final_mod, file = "output/final_mod.Rdata")

predict(final_mod, new_data = IFX_validation[1, ], type = "prob")

# ---------------------------------------- model fairness

rf_validation_predictions <- IFX_validation %>%
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

metrics_by_age <- rf_validation_predictions %>%
    group_by(Montreal_age) %>%
    metrics_result(truth = group, estimate = .pred_class, .pred_0) %>%
    pivot_wider(
        names_from = ".metric",
        values_from = ".estimate"
    ) %>%
    rename("group" = Montreal_age)

## --------------------------------------- gender

metrics_by_gender <- rf_validation_predictions %>%
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

metrics_by_CDAI <- rf_validation_predictions %>%
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
            count(rf_validation_predictions, Montreal_age) %>% select(n),
            count(rf_validation_predictions, gender) %>% select(n),
            count(rf_validation_predictions, CDAI_group) %>% select(n)
        ),
        `Non-response` = bind_rows(
            count(rf_validation_predictions, Montreal_age, group) %>% filter(group == 0) %>% select(n),
            count(rf_validation_predictions, gender, group) %>% filter(group == 0) %>% select(n),
            count(rf_validation_predictions, CDAI_group, group) %>% filter(group == 0) %>% select(n),
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
