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

#### 1 ####

# testing
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
  
  print(paste("For digit", other_digit, "the accuracy is:",
              optimal_parameters$optimal_accuracy))
  print(paste("For digit", other_digit, "the best value for C is:",
              optimal_parameters$selected_parameters$C,
              "and the best value for sigma is:",
              optimal_parameters$selected_parameters$sigma))
  print("-------------")
  
  accuracies[as.character(other_digit)] = optimal_parameters$optimal_accuracy
}


print(accuracies)

least_similar = accuracies[which.max(accuracies)]
most_similar = accuracies[which.min(accuracies)]

print(paste("Least similar:", least_similar))
print(paste("Most similar:", most_similar))
# TODO: try to see if we can make it recognise multiple maxima (low priority)


#### 2. ####  ##TODO: probeer 5 hieruit te halen
svms = list()
accuracies_ij = list()

for (i in 0:8){
  for (j in (i+1):9){
    
    digit_combo = paste(i, "_", j)
    svms[digit_combo] = my_svm(i, j, train, num_samples = 1000,
                               run_grid_search = TRUE, c_vector = c_vector, #heb de grid_search nu even op TRUE gezet 
                               sigma_vector = sigma_vector)$svm 
    
    ## ---- ik heb dit hieronder (tot de volgende ---) toegevoegd, 
    #           waarschijnlijk zal ik het wel weer verkeerd begrepen hebben haha,
    #           maar ik dacht dat dit mogelijk een oplossing kan zijn.
    
    print(paste("svm created for", i, "and", j))
    print(paste("For digit", i, "and", j, "the accuracy is:",
                optimal_parameters$optimal_accuracy))
    print(paste("For digit", i, "_", j, "the best value for C is:",
                optimal_parameters$selected_parameters$C,
                "and the best value for sigma is:",
                optimal_parameters$selected_parameters$sigma))
    print("-------------")
    
    accuracies_ij[as.character(digit_combo)] = optimal_parameters$optimal_accuracy
  }
}

num_rows = dim(test)[1]
labels = test$label
for (row in 1:num_rows){
  label = labels[row]
  for (svm in svms){
    predict()
  }
}
test

sum_accuracies_ij = do.call(sum, accuracies_ij)
acc_mvt = sum_accuracies_ij/length(accuracies_ij)
print(paste("The accuracy of the Majority Vote System is:", acc_mvt))

## -----


num_rows = dim(test)[1]

labels = test$label

svms_test = list()

svms_test["0 _ 1"] =  my_svm(0, 1, train, num_samples = 1000, 
                           c_vector = 0.1, 
                           sigma_vector = sigma_vector)$svm

svms_test["0 _ 2"] =  my_svm(0, 2, train, num_samples = 1000, 
                           c_vector = 0.1, 
                           sigma_vector = sigma_vector)$svm

svms_test["3 _ 4"] =  my_svm(3, 4, train, num_samples = 1000, 
                           c_vector = 0.1, 
                           sigma_vector = sigma_vector)$svm



# TODO: Spaties?
preds = list()

for (dc in c("0 _ 1", "0 _ 2", "3 _ 4")) {
  
  prediction = kernlab::predict(svms_test[[dc]], test[, -1],
                                type="response")
  prediction[prediction==0] = substr(dc, 1, 1)
  prediction[prediction==1] = substr(dc, nchar(dc), nchar(dc))
  
  preds[[dc]] = prediction
}

preds = list()

for (i in 0:8) {
  for (j in (i + 1):9) {
    digit_combo = paste(i, "_", j)
    
    prediction = kernlab::predict(svms[[digit_combo]], test[, -1],
                                  type="response")
    # Als er uit pred een 0 komt, is het de eerste digit en anders de tweede
    prediction[prediction==0] = i
    prediction[prediction==1] = j
    
    preds[digit_combo] = prediction
  }
}

preds_matrix = matrix(unlist(preds), ncol = 10000, byrow=TRUE)

num_cols = length(preds_matrix)

winners = c()
for (col in 1:num_cols) {
  winner = names(which.max(table(preds_matrix[, 1])))
  winners = c(winners, winner)
}

# tapply gebruiken?
boolean_test = (winners == labels)
acc_mvs = sum(boolean_test) / dim(test)[1]




#### 3 ####
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

tvs_nn <- train_validation_split(train, num_samples = 1000, training_size = 0.75)
nn_traindata <- tvs_nn$train
ann_traindata <- nn_traindata[-1]
digits = 0:9

for (H in seq(5,20,5)){
  
  ann <- neuralnet(digits~01+02, data = nn_traindata, hidden = H, act.fct = "logistic",
                   linear.output = FALSE)
  
  plot(ann)
}
