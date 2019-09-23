## Import data
rm(list = ls())

library(caret)
library(kernlab)
library(e1071)

import_data = function(path){
  data = read.csv(path, stringsAsFactors = F, header = F)
  names(data)[1] <- "label"
  data$label <- factor(data$label)
  return(data)
}

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
  
  
  return(tvs)
}
  

# TODO: Change name
my_svm = function(digit1, digit2, train, num_samples, run_grid_search = FALSE,
                  c_vector = 10^(-3:3), sigma_vector = 10^(-8:2)) {
  
  tvs = create_training_set(digit1, digit2, train, num_samples)
  
  train = tvs$train
  validation = tvs$validation
  
  if (run_grid_search){
    results = grid_search(train, validation, c_vector, sigma_vector)
  } else {
    c_scalar = c_vector[1]
    sigma = sigma_vector[1]
    
    svm = kernlab::ksvm(label ~ ., data = train, scaled = F,
                                   kernel = "rbfdot",
                                   C = c_scalar,
                                   kpar = list(sigma = sigma), prob.model=TRUE)
    
    results = list("optimal_accuracy" = NA, 
                   "selected_parameters" = list("C" = c_scalar, 
                                                "sigma" = sigma),
                   "svm" = svm)
  }
  
  return(results)
}

#### Run functions ####

train = import_data("data/mnist_train.csv")
test = import_data("data/mnist_test.csv")

# testing
c_vector = 10^(-2:2)
sigma_vector = 10^(-4:1)

accuracies = list()

for (other_digit in 0:9) {
  if (other_digit == 5) {
    next
  }
  
  optimal_parameters = my_svm(5, other_digit, train, num_samples = 1000,
                              run_grid_search = TRUE, c_vector = c_vector,
                              sigma_vector = sigma_vector)
  
  print(paste("For digit", other_digit, "the accuracy is:", optimal_parameters$optimal_accuracy))
  print("-------------")
  
  accuracies[as.character(other_digit)] = optimal_parameters$optimal_accuracy
}

print(accuracies)

least_similar = accuracies[which.max(accuracies)]
most_similar = accuracies[which.min(accuracies)]

print(paste("Least similar:", least_similar))
print(paste("Most similar:", most_similar))
# TODO: try to see if we can make it recognise multiple maxima (low priority)

#### 2. ####
svms = list()

for (i in 0:8){
  for (j in (i+1):9){
    
    digit_combo = paste(i, "_",j)
    svms[digit_combo] = my_svm(i, j, train, num_samples = 1000, 
                               c_vector = c_vector, 
    
    print(paste("svm created for", i, "and", j))
  }
}



num_rows = dim(test)[1]

labels = test$label


for (i in 0:8) {
  for (j in (i+1):9) {
    
    pixels = test[row, -1]
    digit_combo = paste(i, "_",j)
    
    svm_test = 3 # TODO
    
    kernlab::predict(svm_test, test[, -1], type="response")
            
  }
}

kernlab::predict(svm_test, test[, -1], type="response")
svm_test = my_svm(0, 1, train, num_samples = 1000, 
                  c_vector = 0.1, 
                  sigma_vector = sigma_vector)$svm


