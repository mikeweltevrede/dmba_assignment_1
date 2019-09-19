## Import data
# rm(list = ls())

library(caret)
library(kernlab)
library(e1071)

import_data = function(path){
  data = read.csv(path, stringsAsFactors = F, header = F)
  names(data)[1] <- "label"
  data$label <- factor(data$label)
  return(data)
}

## Exercise 2
train_validation_split = function(train_all, num_samples, training_size=0.75) {
  
  #set.seed(42)
  sample_indices <- sample(1:nrow(train_all), num_samples)
  training_samples = floor(num_samples*training_size)
  
  train <- train_all[sample_indices[1:training_samples], ]
  validation <- train_all[sample_indices[(training_samples + 1):num_samples], ]
  
  return(list('train' = train, 'validation' = validation))
}

grid_search = function(train, validation,
                       kernels = c("vanilladot", "rbfdot", "polydot"),
                       c_vector = 10^(-3:3), sigma_vector = 10^(-8:2),
                       degree_vector = 1:4) {
  
  optimal_accuracy = 0
  selected_parameters = list()
  
  for (c_scalar in c_vector) {
    print(paste("C:", c_scalar))
    
    # Grid Search for Vanilla
    if ("vanilladot" %in% kernels) {
      svm_lin = kernlab::ksvm(label ~ ., data = train, scaled = FALSE,
                              kernel = "vanilladot", C = c_scalar)
      evaluation_lin = predict(svm_lin, newdata = validation, type = "response")
      cm_lin = caret::confusionMatrix(evaluation_lin, validation$label)
      
      current_accuracy = cm_lin$overall["Accuracy"]
      
      if (current_accuracy > optimal_accuracy) {
        optimal_accuracy = current_accuracy
        selected_parameters["C"] = c_scalar
      }
    }
      
    
    
    # Check for RBF
    if ("rbfdot" %in% kernels) {
      for (sigma in sigma_vector) {
        print(paste("Sigma:", sigma))
        
        svm_rbf = kernlab::ksvm(label ~ ., data = train, scaled = F,
                                kernel = "rbfdot", C = c_scalar,
                                kpar = list(sigma = sigma))
        evaluation_rbf = predict(svm_rbf, newdata = validation, type = "response")
        cm_rbf = caret::confusionMatrix(evaluation_rbf, validation$label)
        
        current_accuracy = cm_rbf$overall["Accuracy"]
        
        if (current_accuracy > optimal_accuracy) {
          optimal_accuracy = current_accuracy
          selected_parameters["C"] = c_scalar
          selected_parameters["sigma"] = sigma
        }
      }
    }
      
      
    
    # Check for polynomial
    if ("polydot" %in% kernels) {
      for (degree in degree_vector) {
        print(paste("degree:", degree))
        
        svm_poly = kernlab::ksvm(label ~ ., data = train, scaled = F,
                                 kernel = "polydot", C = c_scalar,
                                 kpar = list(degree = degree))
        evaluation_poly = predict(svm_poly, newdata = validation, type = "response")
        cm_poly = caret::confusionMatrix(evaluation_poly, validation$label)
        
        current_accuracy = cm_poly$overall["Accuracy"]
        
        if (current_accuracy > optimal_accuracy) {
          optimal_accuracy = current_accuracy
          selected_parameters["C"] = c_scalar
          selected_parameters["degree"] = degree
          
        }
      }  
    }
  }
  
  return(list("optimal_accuracy" = optimal_accuracy,
              "selected_parameters" = selected_parameters))
}

# TODO: Change name
my_svm = function(digit1, digit2, train, num_samples,
                  kernels = c("vanilladot", "rbfdot", "polydot"),
                  c_vector = 10^(-3:3), sigma_vector = 10^(-8:2),
                  degree_vector = 1:4) {
  idx_d1 <- which(train$label == digit1)
  idx_d2 <- which(train$label == digit2)
  
  train_all = train[c(idx_d1,idx_d2), ]
  train_all$label <- factor(train_all$label) # Reset factor levels
  
  tvs = train_validation_split(train_all, num_samples)
  train = tvs$train
  validation = tvs$validation
  
  results = grid_search(train, validation, kernels, c_vector, sigma_vector,
                        d_vector)
  
  return(results)
}

#### Run functions ####

# train = import_data("data/mnist_train.csv")
# test = import_data("data/mnist_test.csv")

# testing
c_vector = 10^(-2:2)
sigma_vector = 10^(-4:1)
degree_vector = 1:3

accuracies = list()

for (other_digit in 0:9) {
  
  if (other_digit == 5) {
    next
  }
  
  optimal_parameters = my_svm(5, other_digit, train, num_samples = 1000,
                              kernels = c('rbfdot'), c_vector = c_vector,
                              sigma_vector = sigma_vector,
                              degree_vector = degree_vector)
  
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













