################################################
# SECTION 1: Create edx set and validation set #
################################################

# Load required packages

library(tidyverse)
library(caret)
library(data.table)



# MovieLens 10M dataset:
# > https://grouplens.org/datasets/movielens/10m/
# > http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Download the zipped file, unzip, and load both ratings and movies data
dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Clean up the classes for the movies data frame
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(movieId),
         title = as.character(title),
         genres = as.character(genres))

# Left join the ratings with movies using their movie ID 
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove extraneous data from environment 
rm(dl, ratings, movies, test_index, temp, movielens, removed)





######################################
# SECTION 1: Introduction / Overview #
######################################

# This section describes the data set and summarizes the goal of the project and key steps that were performed

# Basic premise is to train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set

# Basic premise is: 
# - use my own variables to generate a model on the 
# - use average movie rating + averag user rating + rating month, 
# - use Metrics::rmse(validation$rating, predicted values)
# - adjust until RMSE is below threshold for full points


#################################
# SECTION 2: Methods / Analysis #
#################################

# This section explains the process and techniques used, including data cleaning, data exploration and visualization, 
# insights gained, and your modeling approach


# Partition the data in to a test set with 20% and train set with 80%
# Set seed to 92 for reproducing results  
set.seed(92, sample.kind = "Rounding")
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
test_set <- edx %>% slice(test_index)
train_set <- edx %>% slice(-test_index)
rm(test_index, edx)


### ACTUAL ###


# Method #1: NAIVE MODEL
# Start off with a naive model that uses the average rating to predict movie ratings
mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(pred = mu_hat,
                   obs = test_set$rating)

# Method #2: Mean + Average Movie Rating
bi <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

pred_bi <- mu_hat + test_set %>%
  left_join(bi, by = "movieId") %>%
  pull(b_i)

model1_rmse <- RMSE(pred = pred_bi, 
                    obs = test_set$rating, 
                    na.rm = TRUE)
rm(pred_bi)

# Method #3: Mean + Average Movie Rating + Average User Rating
bu <- train_set %>%
  left_join(bi, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

pred_bu <- test_set %>%
  left_join(bi, by = "movieId") %>%
  left_join(bu, by = "userId") %>%
  mutate(pred_bu = mu_hat + b_i + b_u) %>%
  pull(pred_bu)

model2_rmse <- RMSE(pred = pred_bu, 
                    obs = test_set$rating, 
                    na.rm = TRUE)
rm(pred_bu)



# Create and view a summary table
results_tbl <- tibble(
  Method = c("Method #1", "Method #2", "Method #3"),
  Model = c("Naive Model", "Mean + Movie", "Mean + Movie + User"),
  RMSE = c(naive_rmse, model1_rmse, model2_rmse)) %>%
  mutate(`Estimated Points` = case_when(
    RMSE >= 0.90000 ~ 5, 
    RMSE >= 0.86550 & RMSE <= 0.89999 ~ 10,
    RMSE >= 0.86500 & RMSE <= 0.86549 ~ 15,
    RMSE >= 0.86490 & RMSE <= 0.86499 ~ 20,
    RMSE < 0.86490 ~ 25)) %>%
  knitr::kable()



# NEXT STEPS - Regularization and Matrix Factorization
# Look at PH125.8x Section 6: Model Fitting and Recommendation Systems / 6.3: Regularization
# Text book Regularization chapter https://rafalab.github.io/dsbook/large-datasets.html#regularization


### Regularization ###



### Matrix factorization ###

library(recosystem)
set.seed(92, sample.kind = "Rounding")

# Convert the train and test sets into recosystem input format
train_data <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId,
                                           rating = rating,
                                           date = date))

test_data  <-  with(test_set,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating = rating,
                                           date = date))

validation_data  <-  with(validation,  data_memory(user_index = userId, 
                                                   item_index = movieId, 
                                                   rating = rating,
                                                   date = date))

# Create the model object
r <-  recosystem::Reco()

# Select the best tuning parameters. I used the parameters that people has been using wirh Reco() examples such as Qiu(2020)
opts <- r$tune(train_data, 
               opts = list(dim = c(10, 20, 30),          # dim is number of factors 
                           lrate = c(0.1, 0.2),          # learning rate
                           costp_l2 = c(0.01, 0.1),      #regularization for P factors 
                           costq_l2 = c(0.01, 0.1),      # regularization for  Q factors 
                           nthread  = 4, niter = 10))    #convergence can be controlled by a number of iterations (niter) and learning rate (lrate)

# Train the algorithm  
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))


# Calculate the predicted values using Reco test_data   
y_hat_reco <-  r$predict(test_data, out_memory())  #out_memory(): Result should be returned as R objects


RMSE_reco <- RMSE(test_set$rating, y_hat_reco)


# Calculate the predicted values using Reco validation_data  
v_y_hat_reco <-  r$predict(validation_data, out_memory()) #out_memory(): Result should be returned as R objects
head(v_y_hat_reco, 10)

v_RMSE_reco <- RMSE(validation$rating, v_y_hat_reco)













### TEST AREA ###


# Further cleaning on the edx data set, adding variables for creating the model
train_set <- train_set %>%
  # Use timestamp column to define the date and year
  mutate(rating_date = lubridate::as_datetime(timestamp),
         rating_year = lubridate::year(rating_date), 
         release_year = as.double(gsub("[\\(\\)]", "", regmatches(title, gregexpr("\\(.*?\\)", title))[[1]])))


# Find average rating for train set
mu <- mean(train_set$rating)

# To find total number of ratings and average movie rating for each movie
movie_add <- train_set %>% 
  group_by(movieId) %>% 
  summarize(movie_effect = mean(rating - mu))


# To find each user's average given rating
user_add <- train_set %>% 
  group_by(userId) %>% 
  summarize(user_effect = mean(rating))


# Find average rating by individual genre and select top 8 highest rated genres
genre_rating <- train_set %>%
  mutate(genre_rating = strsplit(genres, "|", fixed = TRUE)) %>%
  as_tibble() %>%
  select(rating, genre_rating) %>%
  unnest(genre_rating) %>%
  group_by(genre_rating) %>%
  summarize(avg_genre_rating = mean(rating)) 


genre_add <- left_join(train_set %>%
                         group_by(movieId) %>% 
                         mutate(genre_rating = strsplit(genres, "|", fixed = TRUE)) %>%
                         as_tibble() %>%
                         select(movieId, genre_rating) %>%
                         unnest(genre_rating), 
                       genre_rating, 
                       by = "genre_rating") %>%
  group_by(movieId) %>%
  summarize(avg_genre_rating = mean(avg_genre_rating)) 


# Combine all new fields
test_set$mu <- mean(train_set$rating)
train_set <- left_join(train_set, movie_add, by = "movieId") 
train_set <- left_join(train_set, user_add, by = "userId")
train_set <- left_join(train_set, genre_add, by = "movieId") 


# Examine correlation of all variables using a corrplot
# We can see that movie rating, user rating, number of ratings, and number of top genres have the largest positive correlation with rating
corrplot::corrplot(train_set %>%
                     select(rating, rating_year, release_year, n_ratings, 
                            avg_movie_rating, avg_user_rating, avg_genre_rating) %>%
                     cor(), 
                   method = "number")







######################
# SECTION 3: Results #
######################

# This section presents the modeling results and discusses the model performance




#########################
# SECTION 4: Conclusion #
#########################

# This section gives a brief summary of the report, its limitations and future work