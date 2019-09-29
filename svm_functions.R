import_data = function(path) {
  
  data = read.csv(path, stringsAsFactors = F, header = F)
  names(data)[1] <- "label"
  data$label <- factor(data$label)
  return(data)
}

## Functions
train_validation_split = function(train_all, num_samples, training_size=0.75) {
  
  set.seed(42)
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
    # If grid search is not run, take predefined parameters
    c_scalar = c_vector[1]
    sigma = sigma_vector[1]
    
    svm = kernlab::ksvm(label ~ ., data = train, scaled = F, kernel = "rbfdot",
                        C = c_scalar, kpar = list(sigma = sigma))
    
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

create_v = function(data) {
  
  # Create v matrix
  v = c()
  for (row in 1:dim(data)[1]) {
    v_i = rep(0, 10)
    v_i[data$label[row]] = 1
    v = c(v, v_i)
  }
  
  v_matrix = matrix(v, ncol = 10, byrow = TRUE)
  return(v_matrix)
}

create_u = function(mvs) {
  
  u_matrix = mvs$predictions
  unique_digits = u_matrix %>%
    apply(1, unique) %>%
    apply(2, sort)
  
  # Create u matrix
  for (i in 1:dim(u_matrix)[1]) {
    u_matrix[i, ] = u_matrix[i, ] %>%
      replace(. == unique_digits[1, i], "0") %>%
      replace(. == unique_digits[2, i], "1")
  }
  
  class(u_matrix) <- "numeric"
  return(t(u_matrix))
}
