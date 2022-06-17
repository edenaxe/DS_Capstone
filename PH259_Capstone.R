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

# Download the zipped file, unzip, and load both ratins and movies data
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


# Further cleaning on the edx data set - USING A SUBSET OF EDX DATA SET FOR TESTING PURPOSES
edx_5K <- edx[1:5000,] %>%
  # Use timestamp column to define the date, year, and month
  mutate(rating_date = lubridate::as_datetime(timestamp),
         rating_year = lubridate::year(rating_date), 
         rating_month = lubridate::month(rating_date),
         release_year = as.double(gsub("[\\(\\)]", "", regmatches(title, gregexpr("\\(.*?\\)", title))[[1]])),
         rating_gap = rating_year-release_year,
         movie_age = 2022-release_year)




# To find total number of ratings and average movie rating for each movie
movie_add <- edx_5K %>% 
  group_by(movieId) %>% 
  summarize(n_ratings = n(),
            avg_movie_rating = mean(rating))

# To find each user's average given rating
user_add <- edx_5K %>% 
  group_by(userId) %>% 
  summarize(avg_user_rating = mean(rating))

# Combine all new fields
edx_5K <- left_join(edx_5K, movie_add, by = "movieId") 
edx_5K <- left_join(edx_5K, user_add, by = "userId")


head(edx_5K)

corrplot::corrplot(edx_5K %>%
                     select(rating, rating_year, rating_month, rating_gap, release_year, 
                            movie_age, n_ratings, avg_movie_rating, avg_user_rating) %>%
                     cor())


######################################
# SECTION 1: Introduction / Overview #
######################################

# This section describes the data set and summarizes the goal of the project and key steps that were performed

# Basic premise is to train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set


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




# To Do
# timestamp to year
# Extract movie release date - movies with larger rating periods are "classic"
# genres to seperate columns? to list like starwars tibble? 
# add column that shows total number of ratings, this can be a metric as well
### hypothesis = if movie is in a "serious" genre, larger time span for ratings, and has more ratings it is more likely to have higher rating
# 25 points: RMSE < 0.86490

train_glm <- train(rating ~ factor(genres) + factor(), method = "glm", data = train_set[1:1000,])
train_glm[["results"]][["RMSE"]]


######################
# SECTION 3: Results #
######################

# This section presents the modeling results and discusses the model performance




#########################
# SECTION 4: Conclusion #
#########################

# This section gives a brief summary of the report, its limitations and future work