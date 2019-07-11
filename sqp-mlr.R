######### Preamble #########

options(digits = 4) # Legibility; common sense
set.seed(982) # Reproducibility

# Load libraries needed
library(tidyverse)
library(mlr)

# Note: when installing mlr on your own machine, be sure to 
#   include dependencies = TRUE


######### Data #########

# Read SQP data (some data cleaning has already been performed)
load("sqp.rdata")

# You can View() the data or just print the first few rows:
head(sqp)

# Some algorithms don't like factors, prepare the data so 
#   everythin is dummy coded
sqp_matrix <- model.matrix(~.-1, data = sqp) %>% as.data.frame()

# Divide into training and test (holdout) set
idx_all <- 1:nrow(sqp_matrix) # All row numbers
idx_train <- sample(idx_all, size = 2500) # Only train numbers
idx_test <- idx_all[!idx_all %in% idx_train] # Test is rest

# Define a train and test set by subsetting original data
# Note this also shuffles data, useful for cross-validation later
sqp_matrix_train <- sqp_matrix[idx_train, ]
sqp_matrix_test <- sqp_matrix[idx_test, ]


######### Simple analysis #########

# First, let's train a simple model without tuning on 
#   the whole dataset

# Define task
task <- sqp_matrix_train %>% select(-val) %>%
  makeRegrTask(data = ., target = "rel")

# Define learner
#   For the full list of available learners in mlr, see:
# https://mlr.mlr-org.com/articles/tutorial/integrated_learners.html
lrn <- makeLearner("regr.ranger")

# Fit (train) model
system.time(mod <- train(lrn, task))

# Predict test values
rel_predicted <- predict(mod, newdata = sqp_matrix_test)

# You can now examine how well test data are predicted, 
#   for example:
plot(rel_predicted$data)

# Or on original reliability scale:
plot(rel_predicted$data %>% mutate_all(plogis))

# mlr automatically calculates interesting performance measures
#   for regression, the default is mean squared error (mse)
performance(rel_predicted, measures = rsq)


# To see which other measures are available:
#   https://mlr.mlr-org.com/articles/tutorial/performance.html


######### Full benchmarking experiment #########

# See https://mlr.mlr-org.com/articles/tutorial/benchmark_experiments.html

## Redefine tasks
rel_regr_task <- sqp_matrix_train %>% select(-val) %>%
  makeRegrTask(data = ., target = "rel")

val_regr_task <- sqp_matrix_train %>% select(-rel) %>%
  makeRegrTask(data = ., target = "val")


## Define a search space for each learner's hyperparameter(s)

# To see the hyperparameters that can be tuned for each learner,
#  try getParamSet(). For example:
getParamSet("regr.ranger")
# getParamSet("regr.ksvm") # etc.

# Tuning for Random forests. We're only tuning the number of trees here.
ps_rf <- makeParamSet(
  makeIntegerParam("num.trees", lower = 1L, upper = 1000L)
)

# Tuning for Support vector machine regression
ps_ksvm <- makeParamSet(
  makeNumericParam("sigma", lower = -12, upper = 12,
                   trafo = function(x) 2^x)
)

# Tuning for Gradient boosting regression trees
ps_gbm <- makeParamSet(
  makeIntegerParam("n.trees", lower = 100, upper = 1000),
  makeIntegerParam("interaction.depth", lower = 1, upper = 10),
  makeIntegerParam("n.minobsinnode", lower = 2, upper = 20),
  makeNumericParam("shrinkage", lower = -4, upper = -1,
                   trafo = function(x) 10^x),
  makeNumericParam("bag.fraction", lower = 0.2, upper = 0.8)
)


## Choose a resampling strategy

# Here 10-fold cross-validation is chosen:
rdesc <- makeResampleDesc("CV", iters = 10L)

# To see which other options are available
# (e.g. bootstrap, holdout, blocked/stratified CV, ...):
#   https://mlr.mlr-org.com/articles/tutorial/resample.html

# Choose a performance measure
meas <- rmse

# Choose a tuning method. Here we use random sampling
ctrl <- makeTuneControlRandom(maxit = 10L)

# aside from the standard grid search,
# mlr provides a number of very advanced tuning methods, e.g.
# iterated F-racing, genetic algorithms, DoE based, etc.
# See: https://mlr.mlr-org.com/articles/tutorial/advanced_tune.html


# Combine learners with evaluation strategies

# Support vector regression
tuned_ksvm <- makeTuneWrapper(learner = "regr.ksvm",
                              resampling = rdesc,
                              measures = meas,
                              par.set = ps_ksvm,
                              control = ctrl, show.info = FALSE)

# Random forests from the ranger package
tuned_rf <- makeTuneWrapper(learner = "regr.ranger",
                            resampling = rdesc, measures = meas,
                            par.set = ps_rf, control = ctrl,
                            show.info = FALSE)

# Gradient boosting 
tuned_gbm <- makeTuneWrapper(learner = "regr.gbm",
                             resampling = rdesc, measures = meas,
                             par.set = ps_gbm, control = ctrl,
                             show.info = FALSE)

# Finally, a "fast" knn method
ps_fnn <- makeParamSet(
  makeIntegerParam("k", lower = 1L, upper = 200L)
)

tuned_fnn <- makeTuneWrapper(learner = "regr.fnn",
                             resampling = rdesc, measures = meas,
                             par.set = ps_fnn, control = ctrl,
                             show.info = FALSE)

# Combine all candidate learners into a list
lrns <- list(makeLearner("regr.lm"), # Also linear regression
             tuned_ksvm, tuned_gbm, tuned_rf)

# For demonstration purposes create a list that only 
#    includes random forests and linear regression
#    to speed things up..
faster_lrns <- list(makeLearner("regr.lm"), tuned_fnn)


## Conduct the benchmark experiment

# This can take a very long time!

bmr <- benchmark(
  learners = faster_lrns, # Replace w lrns if you enjoy waiting
  tasks = rel_regr_task,
  resamplings = rdesc,
  # We'll also include some other measures of performance:
  measures = list(rmse, mae, medae, rsq), 
  show.info = TRUE)

## Evaluate performance of various models

# Summarize aggregate (over folds) performances
getBMRAggrPerformances(bmr)

# Also examine distribution (over folds) of performance estimates
plotBMRBoxplots(bmr, measure = rsq)


## Fit final chosen model, and look at importnace, effects

learner_rf <- makeLearner("regr.ranger") %>% 
  setHyperPars(num.trees = 520)

model_rf <- train(learner_rf, rel_regr_task)

# Generate a partial dependence plot for two features
#  see here for more information:
#  https://christophm.github.io/interpretable-ml-book/pdp.html
pd_main <- generatePartialDependenceData(
  model_rf, rel_regr_task, c("fixrefpoints", "ncategories") 
)

# Plot the result using mlr's built-in plotting method:
plotPartialDependence(pd_main)

# Show feature importance for random forest
feature_importance <- getFeatureImportance(model_rf)
feature_importance

