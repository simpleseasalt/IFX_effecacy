# A simple Guide

```R

if (!requireNamespace("randomForest", quietly = TRUE)) {
    install.packages("randomForest")
}

load("final_mod.RData")

newdata <- data.frame(
    Age = 21,
    Weight = 63,
    WBC = 6.74,
    ALB = 45.1,
    CREA = 70.7,
    CDAI_before = 86
)

predict(final_mod, newdata, type = "prob")
```
