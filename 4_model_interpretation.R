# | ---------------------------------------
# | Author: Simplezzz
# | Date: 2024-10-15 21:13:11
# | LastEditTime: 2026-04-28 23:38:53
# | FilePath: \R_scripts\4_model_interpretation.R
# | Description:
# | ---------------------------------------

library(DALEX)
library(DALEXtra)
library(randomForest)
library(tidymodels)
library(probably)
library(stacks)
library(themis)
library(ggthemes)
library(shapviz)
library(kernelshap)
library(cowplot)

load(file = "output/final_mod.RData")

load(file = "output/IFX_validation.RData")

load(file = "output/testset.RData")

load(file = "output/IFX_workspace.RData")

# ---------------------------------------- 

explainer_final <- DALEXtra::explain_tidymodels(
    model = final_mod,
    data = IFX_validation,
    y = IFX_validation$group,
    label = "Random Forest"
)

##---------------------------------------- in dataset level

variable_label <- c(
    "ESR" = "ESR",
    "CDAI_before" = "CDAI",
    "Montreal_age" = "Montreal age",
    "CRP_before" = "CRP",
    "RBC" = "RBC"
)

stack_profile <- model_profile(
    explainer_final,
    variable = names(IFX_train[, c("CRP_before", "CDAI_before", "ESR", "RBC")])
)

plot_profile <- stack_profile %>%
    plot() +
    facet_wrap(
        ~`_vname_`,
        labeller = as_labeller(variable_label),
        scales = "free_x",
        ncol = 2
    ) +
    coord_cartesian(ylim = c(0, 1.0)) +
    scale_y_continuous(breaks = seq(0.0, 1, 0.2)) +
    labs(y = "Probability of Clinical Response") +
    theme_bw() +
    theme(
        title = element_blank(),
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        strip.text = element_text(size = 16),
        panel.spacing = unit(2, "lines"),
        strip.background = element_rect(fill = NA)
    )

plot_profile

tiff(filename = "plot/plot_stack_profile.tiff", width = 10, height = 6, res = 300, units = "in", compression = "lzw")

plot_profile

dev.off()

## --------------------------------------- Montreal age

plot_profile_age <- model_profile(
    explainer_final,
    variable = names(IFX_train[, c("CRP_before", "CDAI_before", "ESR", "RBC")]),
    type = "partial",
    groups = "Montreal_age"
) %>%
    plot(geom = "profiles") +
    facet_wrap(
        ~`_vname_`,
        labeller = as_labeller(variable_label),
        scales = "free_x",
        ncol = 2
    ) +
    coord_cartesian(ylim = c(0, 1.0)) +
    scale_y_continuous(breaks = seq(0.0, 1, 0.2)) +
    scale_color_discrete(
        labels = c(
            "Montreal age (< 16)",
            "Montreal age (16 -40)",
            "Montreal age (≥ 40)"
        ) # 自定义标签
    ) +
    labs(y = "Probability of Clinical Response") +
    theme_bw() +
    theme(
        title = element_blank(),
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        strip.text = element_text(size = 16),
        panel.spacing = unit(2, "lines"),
        legend.position = "bottom",
        legend.text = element_text(size = 14),
        strip.background = element_rect(fill = NA)
    )

plot_profile_age

tiff(filename = "plot/plot_profile_age.tiff", width = 10, height = 10, res = 300, units = "in", compression = "lzw")

plot_profile_age

dev.off()

##---------------------------------------- in person level

example_patient <- predict_parts(explainer_final, new_observation = data_model[, -6][1, ])

plot_explain_patient <- plot(
    example_patient,
    vnames = c("Intercept", "CDAI = 83.4", "RBC = 3.89", "CRP = 55.5", "ESR = 23", "Age (16 - 40)", "Prediction"),
    title = NULL,
    subtitle = NULL,
    digits = 3,
    add_contributions = FALSE
) +
    geom_text(aes(y = right_side), size = 5, color = "black", nudge_y = 0.12) +
    xlab("Probability of Clinical Response") +
    theme(
        title = element_blank(),
        axis.title = element_text(size = 16, face = "bold"),
        axis.text = element_text(size = 14),
        strip.text = element_blank(),
        panel.spacing = unit(0.5, "lines")
    )

plot_explain_patient

tiff(filename = "plot/plot_rf_bd.tiff", width = 10, height = 6, res = 300, units = "in", compression = "lzw")

plot_explain_patient

dev.off()

# ---------------------------------------- shap
## --------------------------------------- importance

X_explain <- IFX_validaton %>%
    select(-group)

calculate_shap <- final_mod %>%
    permshap(X = X_explain, type = "prob") %>%
    shapviz()

shap_value <- calculate_shap$.pred_1

plot_shap_importance <- shap_value %>%
    sv_importance(kind = "beeswarm") +
    theme_bw() +
    theme(
        title = element_blank(),
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        strip.text = element_text(size = 16),
        panel.spacing = unit(2, "lines"),
        legend.position = "right",
        legend.text = element_text(size = 14),
        strip.background = element_rect(fill = NA)
    )

plot_shap_importance

tiff(filename = "plot/shap importance.tiff", width = 10, height = 10, res = 300, units = "in", compression = "lzw")

plot_shap_importance

dev.off()

## --------------------------------------- waterfall

shap_value %>%
    sv_waterfall(
        row_id = 10,
        annotation_size = 6
        ) +
    theme_bw() +
    theme(
        title = element_blank(),
        axis.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 14),
        strip.text = element_text(size = 16),
        panel.spacing = unit(2, "lines"),
        legend.position = "right",
        legend.text = element_text(size = 14),
        strip.background = element_rect(fill = NA)
    )

# ---------------------------------------- combine

plot_interpret <- plot_grid(
    plot_profile,
    plot_explain_patient,
    ncol = 2,
    labels = c("A", "B"), # 可选添加标签
    label_size = 20,
    rel_widths = c(1.2, 0.8) # 按需调整比例
)

tiff(
    filename = "plot/combined_interpret.tiff",
    width = 16, height = 8, res = 300, units = "in", compression = "lzw"
)

print(plot_interpret)

dev.off()

# ! end


