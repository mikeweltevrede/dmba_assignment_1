average_pooling = function(image, stride = 4) {
  
  # Define the height of the image (vector) in matrix form
  height_image = sqrt(length(image))
  
  # We need the image to be square
  if (floor(height_image) != height_image) {
    print(paste("This image is not square. Returning NULL."))
    return(NULL)
  }
  
  # We need the stride to fit exactly into the image size
  if (height_image %% stride != 0) {
    print(paste("This stride is not possible. Please pick a stride in the ",
                "multiplicative table of", sqrt(length(image))))
    return(NULL)
  }
  
  # We could have a mode that is "list". To support proper application of the
  # mean function, we need the image to be a numeric vector instead.
  if (mode(image) != "numeric") {
    storage.mode(image) <- "numeric"
  }
  
  number_of_squares = height_image/stride
  
  image_matrix = matrix(image, nrow = height_image, byrow = TRUE)
  pooled_image = matrix(0, nrow = number_of_squares, ncol = number_of_squares)
  
  boundaries = seq(1, height_image, stride)
  
  for (i in 1:number_of_squares) {
    for (j in 1:number_of_squares) {
      # Pooled image is defined as the average of a stride X stride image
      pooled_image[i, j] = mean(
        image_matrix[boundaries[i]:(boundaries[i] + (stride - 1)),
                     boundaries[j]:(boundaries[j] + (stride - 1))])
    }
  }
  
  # Return image as vector
  pooled_image = as.numeric(pooled_image)
  
  return(pooled_image)
}

random_projection = function(data) {
  return(data)
}
