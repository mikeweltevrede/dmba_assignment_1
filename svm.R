#### Prepare environment ####
rm(list = ls())

library(caret)
library(kernlab)
library(e1071)
library(keras)

#### Import functions ####
source("svm_functions.R")

#### Initialise ####
train = import_data("data/mnist_train.csv")
test = import_data("data/mnist_test.csv")

labels_train = train$label
labels_test = test$label

c_vector = 10^(seq(-2, 2, length.out = 9))
sigma_vector = 10^(seq(-8, -6, length.out = 5))

num_samples = 7500
run_grid_search = FALSE

path_to_svm = "vars/SVMs.RData"
path_to_mvs = "vars/MVS.RData"

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

#### Support Vector Machines ####
if (file.exists(path_to_svm)) {
  load(path_to_svm)
} else if (run_grid_search) {
  svms = create_svms(train, num_samples = num_samples, run_grid_search = TRUE,
                     c_vector = c_vector, sigma_vector = sigma_vector)
  save(svms, file = path_to_svm)
} else{
  svms = create_svms(train, num_samples = num_samples, run_grid_search = FALSE,
                     parameters = parameters)
  
  save(svms, parameters, file = path_to_svm)
}

#### 1. ####
# Consider the digit 5. What is the most similar digit to 5? What is the least
# similar one?

preds = list()

for (dgt in 0:9) {
  if (dgt == 5) {
    next
  }
  
  # Create name to retrieve the corresponding svm
  if (dgt < 5) {
    digit_combo = paste0(dgt, "_", 5)
  } else {
    digit_combo = paste0(5, "_", dgt)
  }
  
  evaluate = test[c(which(test$label == 5), which(test$label == dgt)), ]
  evaluate$label = factor(evaluate$label)
  
  prediction = kernlab::predict(svms[[digit_combo]], test[, -1],
                                type = "response")
  cm = caret::confusionMatrix(predict(svms[[digit_combo]],
                                      newdata = evaluate, type = "response"),
                              evaluate$label)
  print(paste("Accuracy from CM for", digit_combo, ":",
              cm$overall[["Accuracy"]]))
  
  preds[[digit_combo]] = prediction
}

# Turn predictions in matrix for easy comparison of accuracy
preds_matrix = t(matrix(unlist(preds), ncol = dim(test)[1], byrow = TRUE))

for (col in 1:dim(preds_matrix)[2]) {
  
  score = (preds_matrix[, col] == test$label)
  accuracy = sum(score) / dim(test)[1]
  
  digits_sep = strsplit(names(preds)[col], "_")[[1]]
  print(paste("The accuracy of comparing", digits_sep[1], "and", digits_sep[2],
              "is: ", accuracy))
  print("---------")
}

#### Majority Vote ####
# Load file, if available
if (file.exists(path_to_mvs)) {
  load(path_to_mvs)  
} else {
  mvs_train = majority_vote(svms, train)
  mvs = majority_vote(svms, test)
  save(mvs_train, mvs, file = path_to_mvs)
}

#### 2. ####
# What is the accuracy of the majority vote system in this case?
winners = mvs$winners
score = (winners == labels_test)
acc_mvs = sum(score) / dim(test)[1]

print(paste("The accuracy of the MVS is:", acc_mvs))

#### 3. ####
# Looking at each digit separately, which ones have the best/worst predictions?
# What are possible reasons?

accuracies_mvs = list()
for (i in 0:9) {
  accuracy = sum(score[labels_test == i])/length(labels_test[labels_test == i])
  accuracies_mvs[[as.character(i)]] = accuracy
}

print(accuracies_mvs)

#### 4. ####
# Train a neural network with 45 input nodes, one hidden layer with H = 5, 10,
# 15, or 20 nodes in this layer, and 10 output nodes to obtain a voting system.
# For each H what is the accuracy of your prediction when using this system?
# Pick the H performing best.

v_train = create_v(train)
u_train = create_u(mvs_train)

v_test = create_v(test)
u_test = create_u(mvs)

# TODO: Fix this. -> And put to SVM functions
keras_model = function(u_train, v_train, u_test, v_test, h, epochs,
                       batch_size = 32, verbose = 2) {
  
  model = keras_model_sequential() %>%
    layer_dense(
      units = h,
      activation = "relu",
      input_shape = 45
    ) %>%
    layer_dense(
      units = 10,
      activation = "softmax"
    ) %>%
    keras::compile(loss = loss_categorical_crossentropy,
                   optimizer = optimizer_adam(), metrics = c("accuracy"))
  
  history = model %>% fit(
    x = u_train, y = v_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2,
    verbose = verbose, view_metrics = FALSE
  )
  
  accuracy = evaluate(model, x = u_test, y = v_test)$acc
  
  return(list("model" = model, "history" = history, "accuracy" = accuracy))
}

models = list()
for (h in seq.int(5L, 20L, by = 5L)) {
  models[[as.character(h)]] = keras_model(u_train, v_train, u_test, v_test, h,
                                          epochs = 15, batch_size = 64)
}

# Print the accuracies
for (model in models) {
  print(model$history)
  print(model$accuracy)
}

# Unfortunately, plots are not shown in a for-loop
plot(models[["5"]]$history)
plot(models[["10"]]$history)
plot(models[["15"]]$history)
plot(models[["20"]]$history)

# We see that the accuracy for H=15 and H=20 is the best, with 0.9866 and 0.9869
# respectively. We also saw that the models started to overfit at around 12
# epochs, with a batch-size of 32. To make this better, we do a grid search.
H = c(15, 20)
epochs = c(11, 12, 13)
batch_sizes = c(64, 128)

models = list()
for (h in H) {
  for (epoch in epochs) {
    for (batch_size in batch_sizes) {
      print(paste("H:", h, "| Epochs:", epoch, "| Batch size:", batch_size))
      title = paste0(h, "_", epoch, "_", batch_size)
      models[[title]] = keras_model(u_train, v_train, u_test, v_test, h = h,
                                    epochs = epoch, batch_size = batch_size)
    }
  }
}


