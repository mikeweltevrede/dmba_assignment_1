## Import data
rm(list = ls())

library(caret)
library(kernlab)
library(e1071)
library(neuralnet)
library(keras)

import_data = function(path){
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



#### Run functions ####
train = import_data("data/mnist_train.csv")
test = import_data("data/mnist_test.csv")

#### 5 ####
sample_indices <- sample(1:nrow(train), 100)
training_samples = floor(100*0.75)
train1 <- train[sample_indices[1:training_samples], ]
validation1 <- train[sample_indices[(training_samples + 1):100], ]
#ik kreeg de train_validation op de een of andere manier niet voor elkaar

train_x <- train[,2:785] %>% as.matrix()
train_y <- train[,1] %>% 
  keras::to_categorical()

test_x <- validation[,785] %>% as.matrix()



