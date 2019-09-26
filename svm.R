#### Prepare environment ####
rm(list = ls())

library(caret)
library(kernlab)
library(e1071)
library(keras)

#### Define functions ####
import_data = function(path) {
  
  data = read.csv(path, stringsAsFactors = F, header = F)
  names(data)[1] <- "label"
  data$label <- factor(data$label)
  return(data)
}

## Functions
train_validation_split = function(train_all, num_samples, training_size=0.75) {
  
  sample_indices <- sample(1:nrow(train_all), num_samples)
  training_samples = floor(num_samples*training_size)
  
  train <- train_all[sample_indices[1:training_samples], ]
  validation <- train_all[sample_indices[(training_samples + 1):num_samples], ]
  
  return(list('train' = train, 'validation' = validation))
}

grid_search = function(train, validation, c_vector = 10^(-3:3),
                       sigma_vector = 10^(-8:2)) {
  
  optimal_accuracy = 0
  selected_parameters = list()
  
  # Grid Search for RBF
  for (c_scalar in c_vector) {
    print(paste("C:", c_scalar))
    
    for (sigma in sigma_vector) {
      print(paste("Sigma:", sigma))
      
      svm = kernlab::ksvm(label ~ ., data = train, scaled = F,
                          kernel = "rbfdot", C = c_scalar,
                          kpar = list(sigma = sigma))
      evaluation = predict(svm, newdata = validation, type = "response")
      cm = caret::confusionMatrix(evaluation, validation$label)
      
      current_accuracy = cm$overall["Accuracy"]
      
      if (current_accuracy > optimal_accuracy) {
        optimal_accuracy = current_accuracy
        selected_parameters["C"] = c_scalar
        selected_parameters["sigma"] = sigma
        best_svm = svm
      }
    }
  }
  
  return(list("optimal_accuracy" = optimal_accuracy,
              "selected_parameters" = selected_parameters,
              "svm" = best_svm))
}

create_training_set = function(digit1, digit2, train, num_samples){
  idx_d1 <- which(train$label == digit1)
  idx_d2 <- which(train$label == digit2)
  
  train_all = train[c(idx_d1,idx_d2), ]
  train_all$label <- factor(train_all$label) # Reset factor levels
  
  tvs = train_validation_split(train_all, num_samples) 
  
  return(tvs)
}


# TODO: Change name?
my_svm = function(digit1, digit2, train, num_samples, run_grid_search = FALSE,
                  c_vector = 10^(-3:3), sigma_vector = 10^(-8:2)) {
  
  tvs = create_training_set(digit1, digit2, train, num_samples)
  
  train = tvs$train
  validation = tvs$validation
  
  if (run_grid_search) {
    results = grid_search(train, validation, c_vector, sigma_vector)
  } else {
    # If grid search is not run, take the C=10 and sigma=10^-7 (which we note
    # are the most common options, as derived from an earlier run grid search)
    c_scalar = c_vector[1]
    sigma = sigma_vector[1]
    
    svm = kernlab::ksvm(label ~ ., data = train, scaled = F, kernel = "rbfdot",
                        C = c_scalar, kpar = list(sigma = sigma),
                        prob.model = TRUE)
    
    results = list("optimal_accuracy" = NA, 
                   "selected_parameters" = list("C" = c_scalar, 
                                                "sigma" = sigma),
                   "svm" = svm)
  }
  return(results)
}

create_svms = function(train, num_samples, run_grid_search = FALSE,
                       c_vector = c(), sigma_vector = c(),
                       parameters = c()) {
  
  svms = list()
  
  for (i in 0:8) {
    for (j in (i + 1):9) {
      
      print(paste("Creating SVM for", i, "and", j, "..."))
      
      digit_combo = paste0(i, "_", j)
      
      if (!run_grid_search) {
        # Then use predefined parameters
        c_vector = parameters[["C"]][[digit_combo]] # TODO: Check if this works
        sigma_vector = parameters[["sigma"]][[digit_combo]] # TODO: idem
      }
      
      optimal_svm = my_svm(i, j, train, num_samples = num_samples,
                           run_grid_search = run_grid_search,
                           c_vector = c_vector, sigma_vector = sigma_vector)
      svms[digit_combo] = optimal_svm$svm
      
      print(paste("SVM created for", i, "and", j))
      print(paste("Accuracy:", optimal_svm$optimal_accuracy,
                  "| Optimal C:", optimal_svm$selected_parameters$C,
                  "| Optimal sigma:", optimal_svm$selected_parameters$sigma))
      print("-------------")
    }
  }
  return(svms)
}

majority_vote = function(svms, test) {
  
  preds = list()
  
  # This for-loop gets predictions for all 45 SVMs
  for (i in 0:8) {
    for (j in (i + 1):9) {
      
      digit_combo = paste0(i, "_", j)
      
      # kernlab::predict() predicts the entire matrix...
      prediction = kernlab::predict(svms[[digit_combo]], test[, -1],
                                    type = "response")
      
      # Save the result in a list
      preds[[digit_combo]] = prediction
    }
  }
  
  # Unlist these predictions in a matrix for column-wise comparison
  preds_matrix = matrix(unlist(preds), ncol = dim(test)[1], byrow = TRUE)
  
  # Initialise list of winners
  winners = c()
  
  # And retrieve winners from the votes. If there is a tie, the which.max()
  # function takes the digit that was voted for first.
  for (col in 1:dim(preds_matrix)[2]) {
    winner = names(which.max(table(preds_matrix[, col])))
    winners = c(winners, winner)
  }
  
  return(list("predictions" = preds_matrix, "winners" = winners))
}

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
svms = create_svms(train, num_samples = 1000, run_grid_search = FALSE,
                   parameters = parameters)

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
  
  digits = strsplit(names(preds)[col], "_")[[1]]
  print(paste("The accuracy of comparing", digits[1], "and", digits[2],
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
for (digit in 0:9) {
  accuracy = sum(score[labels == digit])/length(labels[labels == digit])
  accuracies_mvs[[as.character(digit)]] = accuracy
}

print(accuracies_mvs)


#### 4. ####
# Train a neural network with 45 input nodes, one hidden layer with H = 5, 10,
# 15, or 20 nodes in this layer, and 10 output nodes to obtain a voting system.
# For each H what is the accuracy of your prediction when using this system?
# Pick the H performing best.

create_u_v = function(data, mvs) {
  
  predictions = t(mvs$predictions)
  labels = data$label
  u_matrix = matrix(as.numeric(predictions == labels), ncol = 45,
                    byrow = TRUE)
  
  v = c()
  for (row in 1:dim(data)[1]) {
    v_i = rep(0, 10)
    v_i[data$label[row]] = 1
    v = c(v, v_i)
  }
  
  v_matrix = matrix(v, ncol = 10, byrow = TRUE)
  return(list("u" = u_matrix, "v" = v_matrix))
}

mvs_train = majority_vote(svms, train)
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

# TODO: Fix this.
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
      activation = "sigmoid"
    ) %>%
    compile(loss = loss_categorical_crossentropy,
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

for (h in H) {
  assign(paste("model_", h), keras_model(u_train, v_train, u_test, v_test, h,
                                         epochs = 30, batch_size = 100))
}

print(model_5$acc)
print(model_10$acc)
print(model_15$acc)
print(model_20$acc)

plot(model_5$hist)
plot(model_10$hist)
plot(model_15$hist)
plot(model_20$hist)

