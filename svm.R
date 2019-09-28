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

c_vector = 10^(-3:3)
sigma_vector = 10^(-8:2)
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

#### Support Vector Machines ####
if (file.exists("vars/SVMs.RData")) {
  load("vars/SVMs.RData")
} else {
  svms = create_svms(train, num_samples = 1000, run_grid_search = FALSE,
                     parameters = parameters)
  save(svms, parameters, "vars/SVMs.RData")
}


#### 1. ####
# Consider the digit 5. What is the most similar digit to 5? What is the least
# similar one?

preds = list()

for (digit in 0:9) {
  if (digit == 5) {
    next
  }
  
  # Create name to retrieve the corresponding svm
  if (digit < 5) {
    digit_combo = paste0(digit, "_", 5)
  } else {
    digit_combo = paste0(5, "_", digit)
  }
  
  prediction = kernlab::predict(svms[[digit_combo]], test[, -1],
                                type = "response")
  preds[[digit_combo]] = prediction
}

# Turn predictions in matrix for easy comparison of accuracy
preds_matrix = t(matrix(unlist(preds), ncol = dim(test)[1], byrow = TRUE))

for (col in 1:dim(preds_matrix)[2]) {
  
  score = (preds_matrix[, col] == data$label)
  accuracy = sum(score) / dim(test)[1]
  
  digits_sep = strsplit(names(preds)[col], "_")[[1]]
  print(paste("The accuracy of comparing", digits_sep[1], "and", digits_sep[2],
              "is: ", accuracy))
  print("---------")
}

#### Majority Vote ####
mvs = majority_vote(svms, test)
predictions = mvs$predictions
winners = mvs$winners

#### 2. ####
# What is the accuracy of the majority vote system in this case?

score = (winners == labels_test)
acc_mvs = sum(score) / dim(test)[1]

print(paste("The accuracy of the MVS is:", acc_mvs))

#### 3. ####
# Looking at each digit separately, which ones have the best/worst predictions?
# What are possible reasons?

accuracies_mvs = list()
for (i in 0:9) {
  accuracy = sum(score[labels == i])/length(labels[labels == i])
  accuracies_mvs[[as.character(i)]] = accuracy
}

print(accuracies_mvs)


#### 4. ####
# Train a neural network with 45 input nodes, one hidden layer with H = 5, 10,
# 15, or 20 nodes in this layer, and 10 output nodes to obtain a voting system.
# For each H what is the accuracy of your prediction when using this system?
# Pick the H performing best.

# Load file, if available
if (file.exists("vars/MVS.RData")) {
  load("vars/MVS.RData")  
} else {
  mvs_train = majority_vote(svms, train)
  save(mvs_train, mvs, file = "vars/MVS.RData")
}

uv = create_u_v(train, mvs_train)
u_train = uv$u
v_train = uv$v

uv = create_u_v(test, mvs)
u_test = uv$u
v_test = uv$v
rm(uv) # Clean up; uv is not needed anymore and only takes up memory

# TODO: Is the interpretation of what u and v are correct? See prints:
print(head(u_test))
print(head(v_test))

# TODO: Fix this. -> And put to SVM functions
keras_model = function(u_train, v_train, u_test, v_test, h, epochs=20,
                       batch_size=50) {
  
  model = keras_model_sequential() %>%
    layer_dense(
      units = h,
      activation = "relu",
      input_shape = 45
    ) %>%
    layer_dense(
      units = 10,
      activation = "relu"
    ) %>%
    keras::compile(loss = loss_categorical_crossentropy,
            optimizer = optimizer_adadelta(), metrics = c("accuracy"))
  
  history = model %>% fit(
    u_train, v_train, verbose = 0,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2
  )
  
  accuracy = evaluate(model, u_test, v_test)$acc
  
  return(list("acc" = accuracy, "hist" = history))
}

H = seq.int(5L, 20L, by = 5L)

models = list()
for (h in H) {
  models[[as.character(h)]] = keras_model(u_train, v_train, u_test, v_test, h,
                                          epochs = 30, batch_size = 100)
}

# Print the accuracies
for (model in models) {
  print(names(model))
  print(model$acc)
}

# Unfortunately, plots are not shown in a for-loop
plot(models[["5"]]$hist)
plot(models[["10"]]$hist)
plot(models[["15"]]$hist)
plot(models[["20"]]$hist)

