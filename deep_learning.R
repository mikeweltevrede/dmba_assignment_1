## Import data
rm(list = ls())

library(caret)
library(kernlab)
library(e1071)
library(neuralnet)

import_data = function(path){
  data = read.csv(path, stringsAsFactors = F, header = F)
  names(data)[1] <- "label"
  data$label <- factor(data$label)
  return(data)
}

## Functions




#### Run functions ####
train = import_data("data/mnist_train.csv")
test = import_data("data/mnist_test.csv")

#### 5 ####

