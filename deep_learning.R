## Import data
rm(list = ls())

library(caret)
library(kernlab)
library(e1071)
library(neuralnet)
library(ANN2)


import_data = function(path){
  data = read.csv(path, stringsAsFactors = F, header = F)
  names(data)[1] <- "label"
  data$label <- factor(data$label)
  return(data)
}

## Functions
train_validation_split = function(train_all, num_samples, training_size = 0.75){
  
  sample_indices <- sample(1:nrow(train_all), num_samples)
  training_samples <- floor(num_samples*training_size)
  
  train <- train_all[sample_indices[1:training_samples], ]
  validation <- train_all[sample_indices[(training_samples + 1):num_samples], ]
  
  return(list('train' = train, 'validation' = validation))
}

  



#### Run functions ####
train = import_data("data/mnist_train.csv")
test = import_data("data/mnist_test.csv")

#### 5 ####
train_auto.data <- train_validation_split(train, num_samples = 1000, training_size = 0.75)
ann784_200 <- neuralnet(autoencoder~0+1+2+3+4+5+6+7+8+9, data = train_auto.data, hidden = 1, learningrate = 0.05, act.fct = "logistic")
plot(ann782_200)

