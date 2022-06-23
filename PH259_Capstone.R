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


# NEXT STEPS - Regularization and Matrix Factorization
# Look at PH125.8x Section 6: Model Fitting and Recommendation Systems / 6.3: Regularization
# Text book Regularization chapter https://rafalab.github.io/dsbook/large-datasets.html#regularization



### Matrix factorization ###

# Follow general process described in recosystem vignette [https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html]
# And https://web.cs.ucdavis.edu/~matloff/matloff/public_html/189/Supplements/Supp02182020.pdf

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

# Step 1. Create a reference class object using Reco()
r <-  Reco()

# Skipped Step 2. data tuning because it was taking a long time and we achieved desired results without it

# Step 3. Train the algorithm, UC Davis paper suggest a rank of 20 as a starting point  
r$train(train_data, opts = list(dim = 20, nmf = TRUE))

# Skipped Step 4. export model via $output()

# Step 5. Calculate the predicted (with $predict()) values using Reco test_data   
pred_MtrxFct <-  r$predict(test_data, out_memory())  #out_memory(): Result should be returned as R objects

MtrxFct_rmse <- RMSE(test_set$rating, pred_MtrxFct)


# Since the matrix factorization gets us where we need to be, repeat the above process but with our validation set
# Convert the validation set into recosystem input format
validation_data  <-  with(validation,  data_memory(user_index = userId, 
                                                   item_index = movieId, 
                                                   rating = rating,
                                                   date = date))

# Calculate the predicted values using Reco validation_data  
pred_MtrxFct_Val <- r$predict(validation_data, out_memory()) #out_memory(): Result should be returned as R objects

MtrxFct_rmse_val <- RMSE(validation$rating, pred_MtrxFct_Val)


######################
# SECTION 3: Results #
######################

# This section presents the modeling results and discusses the model performance


# Create and view a summary table - first 3 models
results_tbl <- tibble(
  Method = c("Method #1", "Method #2", "Method #3"),
  Model = c("Naive Model", "Mean + Movie", "Mean + Movie + User"),
  RMSE = c(naive_rmse, model1_rmse, model2_rmse)) %>%
  mutate(`Estimated Points` = case_when(
    RMSE >= 0.90000 ~ 5, 
    RMSE >= 0.86550 & RMSE <= 0.89999 ~ 10,
    RMSE >= 0.86500 & RMSE <= 0.86549 ~ 15,
    RMSE >= 0.86490 & RMSE <= 0.86499 ~ 20,
    RMSE < 0.86490 ~ 25))

results_tbl %>%
  knitr::kable()


# Redo the results table with matrix factorization 
results_tbl <- tibble(
  Method = c("Method #1", "Method #2", "Method #3", "Method #4"),
  Model = c("Naive Model", "Mean + Movie", "Mean + Movie + User", "Matrix Factorization"),
  RMSE = c(naive_rmse, model1_rmse, model2_rmse, MtrxFct_rmse)) %>%
  mutate(`Estimated Points` = case_when(
    RMSE >= 0.90000 ~ 5, 
    RMSE >= 0.86550 & RMSE <= 0.89999 ~ 10,
    RMSE >= 0.86500 & RMSE <= 0.86549 ~ 15,
    RMSE >= 0.86490 & RMSE <= 0.86499 ~ 20,
    RMSE < 0.86490 ~ 25)) 

results_tbl %>%
  knitr::kable()

# Final results table with test on validation set
# Create a gt() version for final table? 
results_tbl <- tibble(
  Method = c("Method #1", "Method #2", "Method #3", "Method #4", "Final Validation"),
  Model = c("Naive Model", "Mean + Movie", "Mean + Movie + User", "Matrix Factorization", "Matrix Factorization"),
  RMSE = c(naive_rmse, model1_rmse, model2_rmse, MtrxFct_rmse, MtrxFct_rmse_val)) %>%
  mutate(`Estimated Points` = case_when(
    RMSE >= 0.90000 ~ 5, 
    RMSE >= 0.86550 & RMSE <= 0.89999 ~ 10,
    RMSE >= 0.86500 & RMSE <= 0.86549 ~ 15,
    RMSE >= 0.86490 & RMSE <= 0.86499 ~ 20,
    RMSE < 0.86490 ~ 25)) 

results_tbl %>%
  knitr::kable()


#########################
# SECTION 4: Conclusion #
#########################

# This section gives a brief summary of the report, its limitations and future work