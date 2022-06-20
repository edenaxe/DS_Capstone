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



### TEST AREA ###


# Further cleaning on the edx data set, adding variables for creating the model
train_set <- train_set %>%
  # Use timestamp column to define the date and year
  mutate(rating_date = lubridate::as_datetime(timestamp),
         rating_year = lubridate::year(rating_date), 
         release_year = as.double(gsub("[\\(\\)]", "", regmatches(title, gregexpr("\\(.*?\\)", title))[[1]])))


# To find total number of ratings and average movie rating for each movie
movie_add <- train_set %>% 
  group_by(movieId) %>% 
  summarize(n_ratings = n(),
            avg_movie_rating = mean(rating))


# To find each user's average given rating
user_add <- train_set %>% 
  group_by(userId) %>% 
  summarize(avg_user_rating = mean(rating))


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


# To Do
# timestamp to year
# Extract movie release date - movies with larger rating periods are "classic"
# genres to seperate columns? to list like starwars tibble? 
# add column that shows total number of ratings, this can be a metric as well
### hypothesis = if movie is in a "serious" genre, larger time span for ratings, and has more ratings it is more likely to have higher rating
# 25 points: RMSE < 0.86490



### ACTUAL ###


# Method #1: NAIVE MODEL
# Start off with a naive model that uses the average rating to predict movie ratings
mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(pred = mu_hat,
                   obs = test_set$rating)

# Method #2: Mean + Average Movie Rating
# Build a linear model predicting rating from average rating and average movie rating
bi <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

pred_bi <- mu_hat + test_set %>%
  left_join(bi, by = "movieId") %>%
  pull(b_i)

# Three options for RMSE - caret::RMSE, Metrics::rmse, or by hand and remove NAs
#RMSE(pred = predicted_ratings, obs = test_set$rating)
#rmse(actual = test_set$rating, predicted = predicted_ratings)
model1_rmse <- sqrt(mean((pred_bi-test_set$rating)^2, na.rm = TRUE))


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

model2_rmse <- sqrt(mean((pred_bu - test_set$rating)^2, na.rm = TRUE))

# Method #4: Mean + Average Movie Rating + Average User Rating + Average Genre Rating

# DOES NOT WORK # 
test_set <- left_join(test_set, movie_add, by = "movieId") 
test_set <- left_join(test_set, user_add, by = "userId")
test_set <- left_join(test_set, genre_add, by = "movieId") 

lm_fit <- lm(rating ~ avg_movie_rating + avg_user_rating + avg_genre_rating + n_ratings, data = train_set)

predicted_obs <- predict(object = lm_fit,
                         newdata = test_set %>% 
                           select(avg_movie_rating, avg_user_rating, avg_genre_rating, n_ratings))


sqrt(mean((predicted_obs - test_set$rating)^2, na.rm = TRUE))



# Build a results table
results_tbl <- tibble(
  Method = c("Method #1", "Method #2", "Method #3", "Method #4"),
  Model = c("Naive Model", "Mean + Movie", "Mean + Movie + User", "Mean + Movie + User + Genre"),
  RMSE = c(naive_rmse, model1_rmse, model2_rmse, model3_rmse))

results_tbl %>%
  knitr::kable()




######################
# SECTION 3: Results #
######################

# This section presents the modeling results and discusses the model performance




#########################
# SECTION 4: Conclusion #
#########################

# This section gives a brief summary of the report, its limitations and future work