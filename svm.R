## Import data
rm(list = ls())

library(caret)
library(kernlab)
library(e1071)

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


# TODO: Change name
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
    c_scalar = 10
    sigma = 10^(-7)
    
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

#### Run functions ####

train = import_data("data/mnist_train.csv")
test = import_data("data/mnist_test.csv")

#### 1 ####
c_vector = 10^(-3:3)
sigma_vector = 10^(-8:2)

accuracies = list()

#TODO: eventueel nog via een functie doen

for (other_digit in 0:9) {
  if (other_digit == 5) {
    next
  }
  
  optimal_parameters = my_svm(5, other_digit, train, num_samples = 1000,
                              run_grid_search = TRUE, c_vector = c_vector,
                              sigma_vector = sigma_vector)
  
  print(paste("Other digit:", other_digit))
  print(paste("Accuracy:", optimal_parameters$optimal_accuracy,
              "Optimal C:", optimal_parameters$selected_parameters$C,
              "Optimal sigma:", optimal_parameters$selected_parameters$sigma))
  print("-------------")
  
  accuracies[as.character(other_digit)] = optimal_parameters$optimal_accuracy
}

# TODO: try to see if we can make it recognise multiple maxima (low priority)
least_similar = accuracies[which.max(accuracies)]
most_similar = accuracies[which.min(accuracies)]

print(paste("All accuracies:", accuracies))
print(paste("Least similar:", least_similar))
print(paste("Most similar:", most_similar))


#### 2. ####
parameters = list("C" = list("0_1" = , "0_2" = , "0_3" = , "0_4" = ,
                             "0_5" = , "0_6" = , "0_7" = , "0_8" = ,
                             "0_9" = , "1_2" = , "1_3" = , "1_4" = ,
                             "1_5" = , "1_6" = , "1_7" = , "1_8" = ,
                             "1_9" = , "2_3" = , "2_4" = , "2_5" = ,
                             "2_6" = , "2_7" = , "2_8" = , "2_9" = ,
                             "3_4" = , "3_5" = , "3_6" = , "3_7" = ,
                             "3_8" = , "3_9" = , "4_5" = , "4_6" = ,
                             "4_7" = , "4_8" = , "4_9" = , "5_6" = ,
                             "5_7" = , "5_8" = , "5_9" = , "6_7" = ,
                             "6_8" = , "6_9" = , "7_8" = , "7_9" = ,
                             "8_9" = ),
                  "sigma" = list("0_1" = , "0_2" = , "0_3" = , "0_4" = ,
                                 "0_5" = , "0_6" = , "0_7" = , "0_8" = ,
                                 "0_9" = , "1_2" = , "1_3" = , "1_4" = ,
                                 "1_5" = , "1_6" = , "1_7" = , "1_8" = ,
                                 "1_9" = , "2_3" = , "2_4" = , "2_5" = ,
                                 "2_6" = , "2_7" = , "2_8" = , "2_9" = ,
                                 "3_4" = , "3_5" = , "3_6" = , "3_7" = ,
                                 "3_8" = , "3_9" = , "4_5" = , "4_6" = ,
                                 "4_7" = , "4_8" = , "4_9" = , "5_6" = ,
                                 "5_7" = , "5_8" = , "5_9" = , "6_7" = ,
                                 "6_8" = , "6_9" = , "7_8" = , "7_9" = ,
                                 "8_9" = ))

##TODO: probeer 5 hieruit te halen
create_svms = function(train, num_samples, run_grid_search = FALSE,
                       c_vector = c(), sigma_vector = c(),
                       parameters = c()) {
  
  svms = list()
  
  for (i in 0:8) {
    for (j in (i + 1):9) {
      
      print(paste("Creating SVM for", i, "and", j, "..."))
      
      digit_combo = paste0(i, "_", j)
      
      if (!run_grid_search) {
        # Use predefined parameters
        c_vector = parameters[["C"]][digit_combo] # TODO: Check if this works
        sigma_vector = parameters[["sigma"]][digit_combo] # TODO: idem
      }
      
      svms[digit_combo] = my_svm(i, j, train, num_samples = num_samples,
                                 run_grid_search = run_grid_search,
                                 c_vector = c_vector,
                                 sigma_vector = sigma_vector)$svm 
      
      print(paste("SVM created for", i, "and", j))
      print(paste("Accuracy:", optimal_parameters$optimal_accuracy,
                  "Optimal C:", optimal_parameters$selected_parameters$C,
                  "Optimal sigma:", optimal_parameters$selected_parameters$sigma))
      print("-------------")
    }
  }
  return(svms)
}

svms = create_svms(train, num_samples = 1000, run_grid_search = FALSE,
                   parameters = parameters)

## -----

majority_vote = function(svms, test) {

  preds = list()
  
  # This for-loop gets predictions for all 45 SVMs
  for (i in 0:8) {
    for (j in (i + 1):9) {
      
      digit_combo = paste0(i, "_", j)
      
      # kernlab::predict() predicts the entire matrix...
      prediction = kernlab::predict(svms[[digit_combo]], test[, -1],
                                    type = "response")
      
      # and returns a 0 for the first class (i) and a 1 for the second class (j)
      prediction[prediction == 0] = i
      prediction[prediction == 1] = j
      
      # Save the result in a list
      preds[digit_combo] = prediction
    }
  }
  
  # Unlist these predictions in a matrix for column-wise comparison
  preds_matrix = matrix(unlist(preds), ncol = dim(test)[1], byrow = TRUE)
  
  # Initialise list of winners
  winners = c()
  
  # And retrieve winners from the votes. If there is a tie, say between d1 and
  # d2, the which.max() function takes the digit that was voted for first.
  for (col in 1:length(preds_matrix)) {
    winner = names(which.max(table(preds_matrix[, 1])))
    winners = c(winners, winner)
  }

  return(winners)
}

# tapply gebruiken?
winners = majority_vote(svms, test)
score = (winners == test$label)
acc_mvs = sum(score) / dim(test)[1]

#### 3 ####
# TODO
# hier kunnen we toch gewoon de svms[digit_combo]$accuracy pakken?
# Want dan hebben we alle accuracies op een rijtje en kunnen we vervolgens 
# de 3 (bijv.) slechtste en beste pakken toch?


best_pred_acc_ij = data.frame(accuracies_ij)
worst_pred_acc_ij = data.frame(accuracies_ij)

#TODO: fix this for loop
for (i in 1:3){ #print the 3 best prediction (i.e. highest accuracy)
  best_pred <- do.call(max, best_pred_acc_ij)
  print(best_pred) #Here I want to print the combination of digits
  #print(best_pred) #Here I want to print the accuracy of that combination
  new_best_pred_acc_ij = best_pred_acc_ij[best_pred] <- NULL #here I want to remove the best prediction from the loop
  best_pred_acc_ij = new_best_pred_acc_ij
}

##TODO: do the same thing for worst prediction




#### 4 #### 
library(neuralnet)

#TODO krijg hem niet aan de praat

tvs_nn <- train_validation_split(train, num_samples = 1000,
                                 training_size = 0.75)
nn_traindata <- tvs_nn$train
nn_validation <- tvs_nn$validation
rm(tvs_nn)

ann_traindata <- nn_traindata[-1]
digits = 0:9

for (H in seq(5,20,5)) {
  
  ann <- neuralnet::neuralnet(digits ~ 01 + 02, data = nn_traindata, hidden = H,
                              act.fct = "logistic", linear.output = FALSE)
  
  plot(ann)
}
