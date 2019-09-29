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
#split the samples
num_samples = 100
sample_indices <- sample(1:nrow(train), num_samples)
training_samples = floor(num_samples * 0.75)
train1 <- train[sample_indices[1:training_samples], ]
validation1 <- train[sample_indices[(training_samples + 1):num_samples], ]
#ik kreeg de train_validation op de een of andere manier niet voor elkaar

#pre-process
train_x <- train1[,2:785] %>% as.matrix()
train_y <- train[1,1] %>% 
  keras::to_categorical()

test_x <- validation1[,2:785] %>% as.matrix()

#encoder and decoder
input_layer <- 
  layer_input(shape = c(784)) 

encoder <- 
  input_layer %>% 
  layer_dense(units = 200, activation = "relu") %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 45, activation = "relu")  # 45 dimensions for the output layer

decoder <- 
  encoder %>% 
  layer_dense(units = 200, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 784, activation = "relu") # 784 dimensions for the original 4 variables

#create the autoencoder
autoencoder_model <- keras_model(inputs = input_layer, outputs = decoder)

autoencoder_model %>% compile(
  loss='mean_squared_error',
  optimizer='adam',
  metrics = c('accuracy')
)

summary(autoencoder_model)

#train the model onto itself
history <-
  autoencoder_model %>%
  keras::fit(train_x,
             train_x,
             epochs = 100,
             shuffle = TRUE,
             validation_data= list(test_x, test_x)
  )

#check how the training went
plot(history)


#visualize the data
reconstructed_points <- 
  autoencoder_model %>% 
  keras::predict_on_batch(x = train_x)

Viz_data <- 
  dplyr::bind_rows(
    reconstructed_points %>% 
      tibble::as_tibble() %>% 
      setNames(names(train_x %>% tibble::as_tibble())) %>% 
      dplyr::mutate(data_origin = "reconstructed"),
    train_x %>% 
      tibble::as_tibble() %>% 
      dplyr::mutate(data_origin = "original")
  )

Viz_data %>%
  ggplot(aes('Petal.Length','Sepal.Width', color = data_origin))+
  geom_point()

#autoencoder weights
autoencoder_weights <- 
  autoencoder_model %>%
  keras::get_weights()

autoencoder_weights %>% purrr::map_chr(class)

#save the weights
keras::save_model_weights_hdf5(object = autoencoder_model,
                               filepath = 'C:/Users/Daniel/Universiteit_nieuw/Master BAOR/DMBA (Business Analytics)/autoencoder_weights.hdf5',
                               overwrite = TRUE)

#predict
encoder_model <- keras_model(inputs = input_layer, outputs = encoder)

encoder_model %>% keras::load_model_weights_hdf5(filepath = "C:/Users/Daniel/Universiteit_nieuw/Master BAOR/DMBA (Business Analytics)/autoencoder_weights.hdf5",
                                                 skip_mismatch = TRUE,by_name = TRUE)

encoder_model %>% compile(
  loss='mean_squared_error',
  optimizer='adam',
  metrics = c('accuracy')
)

#how did the encoder model do?
#embeded_points <- 
  #encoder_model %>% 
 # keras::predict_on_batch(x = train_x)

#embeded_points %>% head

#measure prediction accuracy
benchmark <- 
  Viz_data_encoded %>%
  mutate(Species = train$Species %>% rep(times = 2)) %>% 
  group_by(data_origin) %>% 
  nest() %>% 
  # mutate(model_lm = data %>% map(glm,formula = Species~., family = binomial())) %>% 
  # mutate(performance = model_lm %>% map(broom::augment)) %>% 
  # unnest(performance,.drop = FALSE)
  mutate(model_caret = data %>% map(~caret::train(form = Species~.,data = .x,method = "rf")))


