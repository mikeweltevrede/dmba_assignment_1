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
library(RandPro)

# Import functions from files
source("svm_functions.R")
source("dimension_reduction_functions.R")

#### Initialise ####
train = import_data("data/mnist_train.csv")
test = import_data("data/mnist_test.csv")

labels_test = data.frame("label" = test$label)

num_samples = 7500

parameters = list(
  "C" = list("0_1" = 10^(-0.5), "0_2" = 10^(0.5), "0_3" = 10^(0.5), "0_4" = 1,
             "0_5" = 1, "0_6" = 10^(0.5), "0_7" = 1, "0_8" = 10^(0.5),
             "0_9" = 1, "1_2" = 1, "1_3" = 1, "1_4" = 1, "1_5" = 10^(1.5),
             "1_6" = 10^(-0.5), "1_7" = 10, "1_8" = 10^(0.5), "1_9" = 10^(0.5),
             "2_3" = 10^(0.5), "2_4" = 10^(1.5), "2_5" = 1, "2_6" = 1,
             "2_7" = 10^(0.5), "2_8" = 10^(0.5), "2_9" = 10, "3_4" = 1,
             "3_5" = 10, "3_6" = 1, "3_7" = 10^(1.5), "3_8" = 10^(0.5),
             "3_9" = 10^(0.5), "4_5" = 1, "4_6" = 10^(0.5), "4_7" = 10^(0.5),
             "4_8" = 10^(0.5), "4_9" = 10, "5_6" = 10^(0.5), "5_7" = 10^(0.5),
             "5_8" = 10^(0.5), "5_9" = 10^(0.5), "6_7" = 1, "6_8" = 10^(0.5),
             "6_9" = 1, "7_8" = 1, "7_9" = 10^(0.5), "8_9" = 10^(0.5)),
  "sigma" = list("0_1" = 10^(-7), "0_2" = 10^(-6.5), "0_3" = 10^(-6.5),
                 "0_4" = 10^(-6.5), "0_5" = 10^(-6.5), "0_6" = 10^(-6.5),
                 "0_7" = 10^(-6.5), "0_8" = 10^(-6.5), "0_9" = 10^(-6.5),
                 "1_2" = 10^(-6.5), "1_3" = 10^(-6.5), "1_4" = 10^(-6.5),
                 "1_5" = 10^(-7), "1_6" = 10^(-6.5), "1_7" = 10^(-6.5),
                 "1_8" = 10^(-6.5), "1_9" = 10^(-6.5), "2_3" = 10^(-6.5),
                 "2_4" = 10^(-7.5), "2_5" = 10^(-6), "2_6" = 10^(-6.5),
                 "2_7" = 10^(-6.5), "2_8" = 10^(-6.5), "2_9" = 10^(-6.5),
                 "3_4" = 10^(-6), "3_5" = 10^(-6.5), "3_6" = 10^(-6.5),
                 "3_7" = 10^(-7), "3_8" = 10^(-6.5), "3_9" = 10^(-6.5),
                 "4_5" = 10^(-6), "4_6" = 10^(-6.5), "4_7" = 10^(-6),
                 "4_8" = 10^(-6.5), "4_9" = 10^(-6.5), "5_6" = 10^(-6.5),
                 "5_7" = 10^(-6.5), "5_8" = 10^(-6.5), "5_9" = 10^(-7),
                 "6_7" = 10^(-6), "6_8" = 10^(-6.5), "6_9" = 10^(-6.5),
                 "7_8" = 10^(-6.5), "7_9" = 10^(-6.5), "8_9" = 10^(-6.5)))

#### Create Random Projection ####
start = proc.time()
train_proj = random_projection(train)
end = proc.time()
time_proj_train = end - start

start = proc.time()
test_proj = random_projection(test)
end = proc.time()
time_proj_test = end - start

#### Create Average Pooling ####
start = proc.time()
train_pooled = train[, -1] %>%
  apply(1, average_pooling) %>%
  t() %>%
  as.data.frame() %>%
  cbind(data.frame("label" = train$label), ., row.names = NULL)
end = proc.time()
time_pool_train = end - start

start = proc.time()
test_pooled = test[, -1] %>%
  apply(1, average_pooling) %>%
  t() %>%
  as.data.frame() %>%
  cbind(data.frame("label" = test$label), ., row.names = NULL)
end = proc.time()
time_pool_test = end - start

#### Create SVMs ####
# Original
start <- proc.time()
svms = create_svms(train, num_samples = num_samples, run_grid_search = FALSE,
                   parameters = parameters)
end <- proc.time()
time_creation_original <- end - start

# Random Projection
start <- proc.time()
svms_proj = create_svms(train_proj, num_samples = num_samples,
                        run_grid_search = FALSE, parameters = parameters)
end <- proc.time()
time_creation_proj = end - start

# Average Pooling
start <- proc.time()
svms_pooled = create_svms(train_pooled, num_samples = num_samples,
                          run_grid_search = FALSE, parameters = parameters)
end <- proc.time()
time_creation_pooled = end - start

#### Majority Voting ####
# Original
start = proc.time()
mvs = majority_vote(svms, test)
winners = mvs$winners

score = (winners == labels_test)
acc_mvs = sum(score) / dim(test)[1]
end = proc.time()
time_mvs_original = end - start

# Random Projection
start = proc.time()
mvs_proj = majority_vote(svms_proj, test_proj)
winners_proj = mvs_proj$winners

score_proj = (winners_proj == labels_test)
acc_mvs_proj = sum(score_proj) / dim(test_proj)[1]
end = proc.time()
time_mvs_proj = end - start

# Average Pooling
start = proc.time()
mvs_pooled = majority_vote(svms_pooled, test_pooled)
winners_pooled = mvs_pooled$winners

score_pooled = (winners_pooled == labels_test)
acc_mvs_pooled = sum(score_pooled) / dim(test_pooled)[1]
end = proc.time()
time_mvs_pooled = end - start

#### Print times ####
print("Times:")
print("Creating randomly projected data:")
print(paste("Train:", time_proj_train["user.self"]))
print(paste("Test:", time_proj_test["user.self"]))

print("Creating average pooled data:")
print(paste("Train:", time_pool_train["user.self"]))
print(paste("Test:", time_pool_test["user.self"]))

print("Creating SVMs:")
print(paste("Original:", time_creation_original["user.self"]))
print(paste("Random Projection:", time_creation_proj["user.self"]))
print(paste("Average Pooling:", time_creation_pooled["user.self"]))

print("Casting and Counting Votes:")
print(paste("Original:", time_mvs_original["user.self"]))
print(paste("Random Projection:", time_mvs_proj["user.self"]))
print(paste("Average Pooling:", time_mvs_pooled["user.self"]))

print("Total time:")
print(paste("Original:", time_creation_original["user.self"]
            + time_mvs_original["user.self"]))
print(paste("Random Projection:", time_proj_train["user.self"]
            + time_proj_test["user.self"]
            + time_creation_proj["user.self"]
            + time_mvs_proj["user.self"]))
print(paste("Average Pooling:", time_pool_train["user.self"]
            + time_pool_test["user.self"]
            + time_creation_pooled["user.self"]
            + time_mvs_pooled["user.self"]))

#### Print Accuracies ####
print("Accuracy MVS:")
print(paste("Original:", acc_mvs))
print(paste("Random Projection:", acc_mvs_proj))
print(paste("Average Pooling:", acc_mvs_pooled))

# Checking individual accuracies looking at each digit separately...
accuracies_mvs = list()
accuracies_mvs_proj = list()
accuracies_mvs_pooled = list()

for (i in 0:9) {
  accuracy = sum(
    score[labels_test == i])/length(labels_test[labels_test == i])
  accuracies_mvs[[as.character(i)]] = accuracy
  
  accuracy = sum(
    score_proj[labels_test == i])/length(labels_test[labels_test == i])
  accuracies_mvs_proj[[as.character(i)]] = accuracy
  
  accuracy = sum(
    score_pooled[labels_test == i])/length(labels_test[labels_test == i])
  accuracies_mvs_pooled[[as.character(i)]] = accuracy
}

print("Individual accuracies:")
accuracies_separate = rbind("Original" = unlist(accuracies_mvs), 
                            "Random Projection" = unlist(accuracies_mvs_proj),
                            "Average Pooling" = unlist(accuracies_mvs_pooled))
print(accuracies_separate)
