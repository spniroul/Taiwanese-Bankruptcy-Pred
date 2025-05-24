# --- Setup ---
set.seed(7)
library(smotefamily)
library(caret)
library(glmnet)
library(xgboost)
library(tidyverse)
library(e1071)
library(yardstick)
library(pROC)
library(randomForest)

# --- Load Data ---
df <- read.csv("data.csv")
df$Bankrupt. <- as.factor(df$Bankrupt.)

# --- Train/Test Split ---
test_idx <- sample(seq_len(nrow(df)), size = 1364)
train <- df[-test_idx, ]
test <- df[test_idx, ]

# --- SMOTE on Training Set ---
smote_result <- SMOTE(X = train[, -1], target = train$Bankrupt., K = 5, dup_size = 15)
train_smote <- smote_result$data
colnames(train_smote)[ncol(train_smote)] <- "Bankrupt."
train_smote$Bankrupt. <- as.factor(train_smote$Bankrupt.)

# --- Elastic Net Variable Selection ---
x_train <- data.matrix(train_smote[, -ncol(train_smote)])
y_train <- train_smote$Bankrupt.

cv.elasticnet <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5, nfolds = 10)
elastic.model <- glmnet(x_train, y_train, family = "binomial", alpha = 0.5, lambda = cv.elasticnet$lambda.1se)

coefs <- coef(elastic.model)
selected_vars <- rownames(coefs)[coefs[, 1] != 0][-1]

# Subset selected variables
train_smote_selected <- train_smote[, c(selected_vars, "Bankrupt.")]
test_selected <- test[, c(selected_vars, "Bankrupt.")]

# --- Logistic Regression (Elastic Net) ---
logit_model <- glmnet(data.matrix(train_smote_selected[, -ncol(train_smote_selected)]),
                      train_smote_selected$Bankrupt., family = "binomial",
                      alpha = 0.5, lambda = cv.elasticnet$lambda.1se)

logit_probs <- predict(logit_model, newx = data.matrix(test_selected[, -ncol(test_selected)]), type = "response")
logit_preds <- ifelse(logit_probs > 0.9, 1, 0)

confusionMatrix(factor(logit_preds), test_selected$Bankrupt., positive = "1")
f2_logit <- f_meas_vec(truth = test_selected$Bankrupt., estimate = factor(logit_preds, levels = c(0, 1)), beta = 2)
print(f2_logit)

# --- XGBoost ---
dtrain <- xgb.DMatrix(data = as.matrix(train_smote_selected[, -ncol(train_smote_selected)]),
                      label = as.numeric(train_smote_selected$Bankrupt.) - 1)
dtest <- xgb.DMatrix(data = as.matrix(test_selected[, -ncol(test_selected)]),
                     label = as.numeric(test_selected$Bankrupt.) - 1)

xgb_model <- xgboost(data = dtrain, objective = "binary:logistic", nrounds = 100,
                     eval_metric = "auc", verbose = 0)

xgb_probs <- predict(xgb_model, dtest)
xgb_preds <- ifelse(xgb_probs > 0.5, 1, 0)

confusionMatrix(factor(xgb_preds), factor(test_selected$Bankrupt.), positive = "1")
f2_xgb <- f_meas_vec(truth = test_selected$Bankrupt., estimate = factor(xgb_preds, levels = c(0, 1)), beta = 2)
print(f2_xgb)

# --- SVM (scaled) ---
svm_model <- svm(Bankrupt. ~ ., data = train_smote_selected, kernel = "radial", cost = 0.05, scale = TRUE)
svm_preds <- predict(svm_model, newdata = test_selected)

confusionMatrix(svm_preds, test_selected$Bankrupt., positive = "1")
f2_svm <- f_meas_vec(truth = test_selected$Bankrupt., estimate = svm_preds, beta = 2)
print(f2_svm)

# --- Random Forest ---
set.seed(42)
rf_model <- randomForest(Bankrupt. ~ ., data = train_smote_selected, ntree = 1000)
rf_preds <- predict(rf_model, newdata = test_selected)

confusionMatrix(rf_preds, test_selected$Bankrupt., positive = "1")
f2_rf <- f_meas_vec(truth = test_selected$Bankrupt., estimate = rf_preds, beta = 2)
print(f2_rf)
