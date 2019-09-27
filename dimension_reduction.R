#### Dimension Reduction ####
# Pick one of the methods used before and evaluate the effect of dimension
# reduction (efficiency gain vs accuracy lost). Consider (separately) at least
# two of the following methods for dimension reduction:
# - PCA (3A)
# - Random projection (3B, 44/5)
# - Projecting each 28x28 image to a 7x7 image by averaging the grey tone in
# each 4x4 sub-square. (6, 43)

# We choose Majority Vote SVM as our method.

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

random_projection = function(data) {
  return(data)
}

average_pooling = function(image, stride = 4) {
  
  # Given a *vector* form image
  
  height_image = sqrt(length(image))
  
  if (floor(height_image) != height_image) {
    print(paste("This image is not square. Returning NULL."))
    return(NULL)
  }
  
  if (height_image %% stride != 0) {
    print(paste("This stride is not possible. Please pick a stride in the ",
                "multiplicative table of", sqrt(length(image))))
    return(NULL)
  }
  
  if (mode(image) != "numeric") {
    storage.mode(image) <- "numeric"
  }
  
  # Given a row in our data of length 784, we want elements
  # (r1/r4, c1/c4), (r5/r8, c5/c8), ..., (r25/r28, c25/c28) but then in vector
  # form: (1/4, 29/32), (1/4, ...)
  
  number_of_squares = height_image/stride
  
  image_matrix = matrix(image, nrow = height_image, byrow = TRUE)
  pooled_image = matrix(0, nrow = number_of_squares, ncol = number_of_squares)
  
  boundaries = seq(1, height_image, stride)
  
  for (i in 1:number_of_squares) {
    for (j in 1:number_of_squares) {
      pooled_image[i, j] = mean(
        image_matrix[boundaries[i]:(boundaries[i] + (stride - 1)),
                     boundaries[j]:(boundaries[j] + (stride - 1))])
    }
  }
  
  # Return image as vector
  pooled_image = as.numeric(pooled_image)
  
  return(pooled_image)
}

train = import_data("data/mnist_train.csv")
test = import_data("data/mnist_test.csv")

a = average_pooling(train[1, -1])

# To debug - shouldn't need to transpose
train_pooled = t(apply(train[1:1000, -1], 1, average_pooling))
