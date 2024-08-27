# | ---------------------------------------
# | Author: Simplezzz
# | Date: 2024-08-06 13:25:18
# | LastEditTime: 2024-08-26 23:29:15
# | FilePath: \script\3_model_construction.R
# | Description: 
# | ---------------------------------------

library(tidyverse)
library(tidymodels)

tidymodels_prefer()

# 1---------------------------------------- 

load("output/final_data.RData")

set.seed(2024)

# 1---------------------------------------- data split

IFX_split <- initial_split(final_data, prop = 0.7, strata = validaty)

IFX_train <- training(IFX_split)

IFX_test <- testing(IFX_split)

IFX_recipe <- recipe(validaty ~ ., data = IFX_train) %>%
    step_scale(all_numeric()) %>%
    step_center() %>%
    step_dummy(all_nominal_predictors())

# 1---------------------------------------- 

mod_plr <- logistic_reg(penalty = tune(), mixture = 1) %>%
    set_engine("glmnet") %>%
    set_mode("classification")

mod_svm <- svm_rbf(cost = tune()) %>%
    set_engine("kernlab") %>%
    set_mode("classification")

mod_rf <- rand_forest(mtry = tune(), min_n = tune(), trees = 2000) %>%
    set_engine("randomForest") %>%
    set_mode("classification")

mod_xgb <- boost_tree(
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
) %>%
    set_engine("xgboost") %>%
    set_mode("classification")

# 2----------------------------------------

IFX_cv <- vfold_cv(IFX_train, strata = "validaty", repeats = 1, v = 10)

# 2---------------------------------------- workflow

IFX_wf <- workflow_set(
    preproc = list(
        recipe = IFX_recipe
    ),
    models = list(
        SVM = mod_svm,
        PLR = mod_plr,
        RF = mod_rf,
        XGB = mod_xgb
    )
) %>%
    mutate(
        wflow_id = case_when(
            wflow_id == "recipe_SVM" ~ "SVM",
            wflow_id == "recipe_PLR" ~ "PLR",
            wflow_id == "recipe_RF" ~ "RF",
            wflow_id == "recipe_XGB" ~ "XGB",
        )
    )

IFX_control <- control_grid(
    allow_par = T,
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
)

grid_result <- workflow_map(
    IFX_wf,
    resamples = IFX_cv,
    metrics = metric_set(roc_auc),
    grid = 20,
    verbose = T,
    seed = 2024,
    control = IFX_control
)

autoplot(
    grid_result,
    rank_metric = "roc_auc"
) +
    theme_bw()

# 2---------------------------------------- update model

PLR_new_para <- parameters(penalty()) %>%
    update(penalty = penalty(c(-9, -1)))

XGB_new_para <- parameters(
    min_n(),
    tree_depth(),
    learn_rate(),
    loss_reduction()
) %>%
    update(
        min_n = min_n(c(1L, 12L)),
        tree_depth = tree_depth(c(1L, 15L)),
        learn_rate = learn_rate(c(-8, -1)),
        loss_reduction = loss_reduction(c(-8, 1.5))
    )

updated_wf <- IFX_wf %>%
    option_add(
        param_info = PLR_new_para,
        id = "PLR"
    ) %>%
    option_add(
        param_info = XGB_new_para,
        id = "XGB"
    ) %>%
    workflow_map(
        resamples = IFX_cv,
        verbose = T,
        control = IFX_control,
        grid = 20,
        seed = 2024,
        metrics = metric_set(roc_auc, accuracy, precision, sensitivity, specificity, recall, f_meas)
    )

updated_wf %>%
    extract_workflow_set_result("XGB") %>%
    unnest(cols = .metrics) %>%
    filter(.metric == "roc_auc") %>%
    arrange(tree_depth) %>%
    view()

updated_mods_plot <- autoplot(
    updated_wf,
    metric = "roc_auc",
    rank_metric = "roc_auc"
) +
    labs(x = "Model Rank", y = "AUROC") +
    scale_shape_discrete(guide = "none") +
    scale_color_discrete(name = "Models", label = c("XGBoost", "PLR", "RF", "SVM")) +
    theme_bw() +
    theme(
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 18, face = "bold"),
        legend.position = "right"
    )

tiff(filename = "plot/updated_mods_plot.tiff", width = 10, height = 6.6, res = 300, units = "in", compression = "lzw")

updated_mods_plot

dev.off()

# 2---------------------------------------- export result
# 3---------------------------------------- glm

PLR_tune <- updated_wf %>%
    extract_workflow_set_result("PLR") %>%
    unnest(cols = .metrics) %>%
    filter(.metric == "roc_auc") %>%
    group_by(penalty) %>%
    summarise(
        mean = mean(.estimate),
        sd = sd(.estimate)
    )

write.csv(PLR_tune, "output/PLR_tune.csv")

# 3---------------------------------------- svm

SVM_tune <- updated_wf %>%
    extract_workflow_set_result("SVM") %>%
    unnest(cols = .metrics) %>%
    filter(.metric == "roc_auc") %>%
    group_by(cost) %>%
    summarise(
        mean = mean(.estimate),
        sd = sd(.estimate)
    )

write.csv(SVM_tune, "output/SVM_tune.csv")

# 3---------------------------------------- rf

RF_tune <- updated_wf %>%
    extract_workflow_set_result("RF") %>%
    unnest(cols = .metrics) %>%
    filter(.metric == "roc_auc") %>%
    group_by(mtry, min_n) %>%
    summarise(
        mean = mean(.estimate),
        sd = sd(.estimate)
    )

write.csv(RF_tune, "output/RF_tune.csv")

# 3---------------------------------------- xgb

XGB_tune <- updated_wf %>%
    extract_workflow_set_result("XGB") %>%
    unnest(cols = .metrics) %>%
    filter(.metric == "roc_auc") %>%
    group_by(min_n, tree_depth, learn_rate, loss_reduction) %>%
    summarise(
        mean = mean(.estimate),
        sd = sd(.estimate)
    )

write.csv(XGB_tune, "output/XGB_tune.csv")

# 2---------------------------------------- plot AUROC

tidymodels_prefer()

updated_result <- updated_wf %>%
    collect_metrics() %>%
    filter(.metric == "roc_auc") %>%
    group_by(wflow_id) %>%
    arrange(-mean) %>%
    slice(1)

updated_result

write_csv(updated_result, "output/updated_result.csv")

updated_wf %>%
    collect_metrics() %>%
    pivot_wider(
        names_from = .metric,
        values_from = c(mean, std_err)
    ) %>%
    group_by(wflow_id) %>%
    arrange(-mean_roc_auc) %>%
    slice(1)

best_mods <- updated_result %>%
    mutate(model_name = paste(.config, model, sep = "_")) %>%
    pull(model_name)

prediction_in_best_mods <- updated_wf %>%
    workflowsets::collect_predictions() %>%
    mutate(model_name = paste(.config, model, sep = "_")) %>%
    filter(
        model_name %in% best_mods
    )

updated_roc_plot <- prediction_in_best_mods %>%
    group_by(wflow_id) %>%
    roc_curve(
        validaty,
        .pred_no
    ) %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity, color = wflow_id)) +
    geom_path(lwd = 1) +
    geom_abline(lty = 3) +
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

tiff(filename = "plot/updated_roc_plot.tiff", width = 10, height = 6.6, res = 300, units = "in", compression = "lzw")

updated_roc_plot

dev.off()

# 1---------------------------------------- fit on test data

PLR_fit <- finalize_workflow(
    extract_workflow(updated_wf, id = "PLR"),
    select_best(
        updated_wf[updated_wf$wflow_id == "PLR", "result"][[1]][[1]],
        metric = "roc_auc"
    )
) %>%
    last_fit(split = IFX_split)

XGB_fit <- finalize_workflow(
    extract_workflow(updated_wf, id = "XGB"),
    select_best(
        updated_wf[updated_wf$wflow_id == "XGB", "result"][[1]][[1]],
        metric = "roc_auc"
    )
) %>%
    last_fit(split = IFX_split)

RF_fit <- finalize_workflow(
    extract_workflow(updated_wf, id = "RF"),
    select_best(
        updated_wf[updated_wf$wflow_id == "RF", "result"][[1]][[1]],
        metric = "roc_auc"
    )
) %>%
    last_fit(split = IFX_split)

SVM_fit <- finalize_workflow(
    extract_workflow(updated_wf, id = "SVM"),
    select_best(
        updated_wf[updated_wf$wflow_id == "SVM", "result"][[1]][[1]],
        metric = "roc_auc"
    )
) %>%
    last_fit(split = IFX_split)

last_fit_list <- list(
    PLR_fit,
    RF_fit,
    SVM_fit,
    XGB_fit
)

last_fit_res <- last_fit_list %>%
    map(collect_metrics) %>%
    rlist::list.stack() %>%
    filter(.metric == "roc_auc") %>%
    mutate(group = c("PLR", "RF", "SVM", "Xgboost"))

data.frame(
    group = updated_result$wflow_id,
    roc_train = round(updated_result$mean, 3),
    roc_test = round(last_fit_res$.estimate, 3)
)

train_fit <- prediction_in_best_mods %>%
    select(c(wflow_id, validaty, .pred_no)) %>%
    mutate(split = "train") %>%
    mutate(
        wflow_id = case_when(
            wflow_id == "PLR" ~ "PLR",
            wflow_id == "XGB" ~ "XGBoost",
            wflow_id == "RF" ~ "RF",
            wflow_id == "SVM" ~ "SVM",
        )
    )

final_roc_plot <- list(PLR_fit, XGB_fit, RF_fit, SVM_fit) %>%
    map(collect_predictions) %>%
    data.table::rbindlist(idcol = "wflow_id") %>%
    mutate(
        wflow_id = case_when(
            wflow_id == 1 ~ "PLR",
            wflow_id == 2 ~ "XGBoost",
            wflow_id == 3 ~ "RF",
            wflow_id == 4 ~ "SVM",
        ),
        split = "test",
    ) %>%
    select(wflow_id, validaty, .pred_no, split) %>%
    rbind(train_fit) %>%
    mutate(group = paste(wflow_id, split, sep = "_")) %>%
    group_by(group) %>%
    roc_curve(
        validaty,
        .pred_no
    ) %>%
    tidyr::separate(group, c("wflow_id", "split"), remove = F) %>%
    mutate(label = case_when(
        wflow_id == "PLR" ~ "AUROC\nTrain 0.843\nTest 0.779",
        wflow_id == "RF" ~ "AUROC\nTrain 0.874\nTest 0.844",
        wflow_id == "SVM" ~ "AUROC\nTrain 0.826\nTest 0.704",
        wflow_id == "XGBoost" ~ "AUROC\nTrain 0.792\nTest 0.812"
    )) %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity, color = split)) +
    geom_path(lwd = 1) +
    geom_abline(lty = 3) +
    facet_wrap(~wflow_id) +
    geom_text(
        aes(x = 0.6, y = 0.25, label = label),
        color = "black",
        size = 5,
        hjust = 0,
        check_overlap = T
    ) +
    coord_equal() +
    theme_bw() +
    labs(x = "1 - Specificity", y = "Sensitivity") +
    scale_color_discrete(name = "Dataset", label = c("Test", "Train")) +
    theme(
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14, face = "bold"),
        strip.text = element_text(size = 12, face = "bold"),
        panel.spacing.x = unit(1, "lines")
    )

tiff(filename = "plot/final_roc_plot.tiff", width = 10, height = 6.6, res = 300, units = "in", compression = "lzw")

final_roc_plot

dev.off()

# 2---------------------------------------- confuse matrix
# 3---------------------------------------- in trainset

train_fit <- prediction_in_best_mods %>%
    select(c(wflow_id, validaty, .pred_no)) %>%
    mutate(split = "train")

train_fit %>%
    mutate(.pred_res = factor(if_else(.pred_no >= 0.5, "no", "yes"))) %>%
    group_by(wflow_id) %>%
    conf_mat(validaty, .pred_res) %>%
    as.data.frame()

# 3---------------------------------------- in testset

last_fit_list %>%
    map_df(collect_predictions) %>%
    mutate(group = c(
        rep("PLR", 64),
        rep("RF", 64),
        rep("SVM", 64),
        rep("Xgboost", 64)
    )) %>%
    group_by(group) %>%
    conf_mat(validaty, .pred_class) %>%
    as.data.frame()

# 1---------------------------------------- model explain

final_mod <- finalize_workflow(
    extract_workflow(updated_wf, id = "RF"),
    select_best(
        updated_wf[updated_wf$wflow_id == "RF", "result"][[1]][[1]],
        metric = "roc_auc"
    )
) %>%
    last_fit(split = IFX_split) %>%
    extract_workflow()

predict(final_mod, new_data = IFX_test[1, ])

save(final_mod, file = "output/final_mod.RData")

# 2---------------------------------------- importance of variables

library(vip)

vip_plot <- final_mod %>%
    extract_fit_parsnip() %>%
    vip::vi() %>%
    mutate(
        label = case_when(
            Variable == "CDAI_before" ~ "CDAI",
            Variable == "ALB" ~ "ALB",
            TRUE ~ Variable
        )
    ) %>%
    ggplot() +
    aes(x = Importance, y = reorder(label, Importance), fill = reorder(label, Importance)) +
    geom_bar(stat = "identity", orientation = "y", width = 0.5) +
    geom_text(aes(label = round(Importance, 1), hjust = -0.5), size = 7) +
    scale_fill_brewer() +
    xlim(c(0, 22)) +
    theme_bw() +
    theme(
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14, face = "bold"),
        strip.text = element_text(size = 14, face = "bold"),
        axis.title.y = element_blank(),
        axis.text.y = element_text(face = "bold", size = 15),
        legend.position = "none"
    )

tiff(filename = "plot/vip_plot.tiff", width = 10, height = 6.6, res = 300, units = "in", compression = "lzw")

vip_plot

dev.off()

# ! end