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
# See exercise 1.3

# First, let's train a simple model without tuning on 
#   the whole dataset

# Define task
task <- sqp_matrix_train %>% select(-val) %>%
  makeRegrTask(data = ., target = "rel")

# Define learner
#   For the full list of available learners in mlr, see:
# https://mlr.mlr-org.com/articles/tutorial/integrated_learners.html
lrn <- makeLearner("regr.lm") 

# Fit (train) model
system.time( # This will show how long it takes, gives an idea of CV timing
  mod <- train(lrn, task)
)

# Predict test values
rel_predicted_test <- predict(mod, newdata = sqp_matrix_test)

# Predict train values
rel_predicted_train <- predict(mod, 
                               newdata = sqp_matrix_train)

# You can now examine how well test data are predicted, 
#   for example:
plot(rel_predicted_test$data)

# Or on original reliability scale:
plot(rel_predicted_test$data %>% mutate_all(plogis))

# mlr automatically calculates interesting performance measures
#   for regression, the default is mean squared error (mse). 
# Here we ask for root-mean squared error and R-squared:
performance(rel_predicted_test, 
            measures = list(rmse, rsq))

performance(rel_predicted_train, 
            measures = list(rmse, rsq))



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
rdesc <- makeResampleDesc("CV", iters = 3L)

# To see which other options are available
# (e.g. bootstrap, holdout, blocked/stratified CV, ...):
#   https://mlr.mlr-org.com/articles/tutorial/resample.html

# Choose a performance measure
meas <- rmse

# Choose a tuning method. Here we use random sampling
ctrl <- makeTuneControlRandom(maxit = 3L)

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
             tuned_fnn, tuned_ksvm, tuned_gbm, tuned_rf)

## Conduct the benchmark experiment

# This can take a very long time!
#  depending on number of CV folds and makeTuneControlRandom maxit
# with k = 3 and maxit = 3 it should take about 120s
system.time(
  bmr <- benchmark(
    learners = lrns,
    tasks = rel_regr_task,
    resamplings = rdesc,
    # We'll also include some other measures of performance:
    measures = list(rmse, mae, medae, rsq), 
    show.info = TRUE)
)

## Evaluate performance of various models

# Summarize aggregate (over folds) performances
getBMRAggrPerformances(bmr)

# Also examine distribution (over folds) of performance estimates
plotBMRBoxplots(bmr, measure = rsq)


####### Tune and train the chosen model ####### 

# After choosing a model, we can examine the performance 
#  in the experiment, or repeat the tuning for all folds
#  to tune the selected model using resampling

# (Note that it is also possible to perform the 
#  model selection and tuning in one step, 
#  using makeModelMultiplexer(). 
#  See: https://mlr.mlr-org.com/reference/makeModelMultiplexer.html)

# This time we'll also tune the mtry parameter
ps_rf_ext <- makeParamSet(
  # With more time choose upper 2000+
  makeIntegerParam("num.trees", lower = 1L, upper = 500L),
  # I've chosen the upper as sqrt(p) but with more time choose p
  makeIntegerParam("mtry", lower = 1L, upper = sqrt(ncol(sqp_matrix)))
)

res <- tuneParams(learner = "regr.ranger",
                  task = rel_regr_task, 
                  # With more time choose 10 or bootstrap
                  resampling = makeResampleDesc("CV", iters = 3L), 
                  measures = rmse,
                  par.set = ps_rf_ext, 
                  # With more time choose genetic algortihm or large maxit
                  control = makeTuneControlRandom(maxit = 5),
                  show.info = TRUE)

# Fit final chosen model with optimal hyperparameters:
learner_rf <- makeLearner("regr.ranger", importance = "permutation") %>% 
  setHyperPars(num.trees = res$x$num.trees, mtry = res$x$mtry)

model_final <- train(learner_rf, rel_regr_task)


###### Model summaries ###### 

# Evaluate final error using hold test data
# If we did not overfit using the cross-validation folds above, 
#  this should give rmse similar to that shown by tuning
predictions_test_final <- predict(model_final, newdata = sqp_matrix_test)
performance(predictions_test_final, measures = list(rmse, rsq))

# Generate a partial dependence plot for two features
#  see here for more information:
#  https://christophm.github.io/interpretable-ml-book/pdp.html
pd_main <- generatePartialDependenceData(
  model_final, rel_regr_task, c("fixrefpoints", "ncategories") 
)

# Plot the result using mlr's built-in plotting method:
plotPartialDependence(pd_main)

# Show feature importance (only for random forest!)
feature_importance <- getFeatureImportance(model_final)
feature_importance

# Below I make a plot of the top-n features in terms of their
#   permutation importance - 
# see https://christophm.github.io/interpretable-ml-book/feature-importance.html
# Note this type of "importance" was criticized by Strobl (2016)

n <- 20  # Top-n 

# Get features and their "importance"
topn <- feature_importance$res %>% 
  sort(decreasing = TRUE) %>% .[1:n] %>% unlist %>% rev

# Adjust plot spacing to longest name of feature
spc <- names(topn) %>% sapply(nchar) %>% max(.)/2
par(mar = c(5, spc, 2, 1))

# Use base R graphics to plot importance
topn %>% 
  plot(., 1:n, pch = 16, col = "steelblue", 
       axes = FALSE, ylab = "", xlab = "Permutation importance",
       main = sprintf("Top %d features", n), xlim=c(0, max(topn)))
axis(2, at = 1:n, labels = names(topn), las = 2, cex.axis = 0.8)
axis(1, at = round(c(min(topn), median(range(topn)), max(topn)), 3))
segments(rep(0, n), 1:n, topn, 1:n, col = "#44444433")
