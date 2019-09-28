#### Dimension Reduction ####
# Pick one of the methods used before and evaluate the effect of dimension
# reduction (efficiency gain vs accuracy lost). Consider (separately) at least
# two of the following methods for dimension reduction:
# - PCA (3A)
# - Random projection (3B, 44/5)
# - Projecting each 28x28 image to a 7x7 image by averaging the grey tone in
# each 4x4 sub-square. (6, 43)

# We choose Majority Vote SVM as our method.

#### Prepare environment ####
rm(list = ls())

library(caret)
library(kernlab)
library(e1071)
library(keras)

#### Define functions ####
source("svm_functions.R")
source("dimension_reduction_functions.R")

#### Initialise ####
train = import_data("data/mnist_train.csv")
test = import_data("data/mnist_test.csv")

labels_train = data.frame("label" = train$label)
labels_test = data.frame("label" = test$label)

#### Average Pooling ####
# To debug - shouldn't need to transpose outcome?
start = proc.time()
train_pooled = train[, -1] %>%
  apply(1, average_pooling) %>%
  t() %>%
  as.data.frame() %>%
  cbind(labels_train, ., row.names = NULL)
end = proc.time()
time_pool_train = end - start

start = proc.time()
test_pooled = test[, -1] %>%
  apply(1, average_pooling) %>%
  t() %>%
  as.data.frame() %>%
  cbind(labels_test, ., row.names = NULL)
end = proc.time()
time_pool_test = end - start

# Create SVMs
parameters = list(
  "C" = list("0_1" = 1, "0_2" = 1, "0_3" = 10, "0_4" = 100, "0_5" = 1,
             "0_6" = 1, "0_7" = 1, "0_8" = 1, "0_9" = 10, "1_2" = 10,
             "1_3" = 0.1, "1_4" = 1, "1_5" = 1, "1_6" = 1, "1_7" = 10,
             "1_8" = 10, "1_9" = 1, "2_3" = 10, "2_4" = 10, "2_5" = 1,
             "2_6" = 1, "2_7" = 10, "2_8" = 1, "2_9" = 10, "3_4" = 1,
             "3_5" = 10, "3_6" = 10, "3_7" = 10, "3_8" = 10, "3_9" = 10,
             "4_5" = 0.1, "4_6" = 0.1, "4_7" = 10, "4_8" = 1, "4_9" = 10,
             "5_6" = 1, "5_7" = 0.1, "5_8" = 1, "5_9" = 10, "6_7" = 0.1,
             "6_8" = 1, "6_9" = 0.1, "7_8" = 1, "7_9" = 1, "8_9" = 1),
  "sigma" = list("0_1" = 10^(-7), "0_2" = 10^(-7), "0_3" = 10^(-7),
                 "0_4" = 10^(-8), "0_5" = 10^(-6), "0_6" = 10^(-7),
                 "0_7" = 10^(-7), "0_8" = 10^(-6), "0_9" = 10^(-7),
                 "1_2" = 10^(-8), "1_3" = 10^(-7), "1_4" = 10^(-7),
                 "1_5" = 10^(-7), "1_6" = 10^(-7), "1_7" = 10^(-7),
                 "1_8" = 10^(-7), "1_9" = 10^(-7), "2_3" = 10^(-7),
                 "2_4" = 10^(-6), "2_5" = 10^(-7), "2_6" = 10^(-7),
                 "2_7" = 10^(-7), "2_8" = 10^(-6), "2_9" = 10^(-8),
                 "3_4" = 10^(-7), "3_5" = 10^(-7), "3_6" = 10^(-7),
                 "3_7" = 10^(-7), "3_8" = 10^(-7), "3_9" = 10^(-7),
                 "4_5" = 10^(-7), "4_6" = 10^(-7), "4_7" = 10^(-7),
                 "4_8" = 10^(-7), "4_9" = 10^(-7), "5_6" = 10^(-6),
                 "5_7" = 10^(-6), "5_8" = 10^(-6), "5_9" = 10^(-7),
                 "6_7" = 10^(-7), "6_8" = 10^(-6), "6_9" = 10^(-7),
                 "7_8" = 10^(-7), "7_9" = 10^(-6), "8_9" = 10^(-7)))

# Start the clock for creating SVMs
# Time original creating function:
start <- proc.time()
svms = create_svms(train, num_samples = 1000, run_grid_search = FALSE,
                   parameters = parameters)
end <- proc.time()

time_creation_original <- end - start

start <- proc.time()
svms_pooled = create_svms(train_pooled, num_samples = 1000,
                          run_grid_search = FALSE, parameters = parameters)
end <- proc.time()

time_creation_pooled = end - start

# Majority Vote
# Original
start = proc.time()
  mvs = majority_vote(svms, test)
  winners = mvs$winners

  score = (winners == labels_test)
  acc_mvs = sum(score) / dim(test)[1]
end = proc.time()
time_mvs_original = end - start

# MVS for pooled
start = proc.time()
  mvs_pooled = majority_vote(svms_pooled, test_pooled)
  winners_pooled = mvs_pooled$winners
  
  score_pooled = (winners == labels_test)
  acc_mvs_pooled = sum(score_pooled) / dim(test_pooled)[1]
end = proc.time()
time_mvs_pooled = end - start

print("Times:")
print("Creating pooled data:")
print(time_pool_train)
print(time_pool_test)

print("Creating SVMs:")
print(time_creation_original)
print(time_creation_pooled)

print("MVS:")
print(time_mvs_original)
print(time_mvs_pooled)

print("Total time:")
print(paste("Original:", time_creation_original["user.self"]
            + time_mvs_original["user.self"]))
print(paste("Pooled:", time_pool_train["user.self"]
            + time_pool_test["user.self"]
            + time_creation_pooled["user.self"]
            + time_mvs_pooled["user.self"]))

print("-----------------")
print("Accuracy:")
print(paste("Original:", acc_mvs))
print(paste("Pooled:", acc_mvs_pooled))


#### Random projection ####

