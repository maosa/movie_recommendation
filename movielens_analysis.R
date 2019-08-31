# Clear the environment and any plots

rm(list = ls())
if(!is.null(dev.list())) { dev.off() }

# Load required packages

library(lubridate)

# Set the working directory

# setwd(file.path(Sys.getenv("HOME"), "Desktop/data_science_HarvardX/movie_recommendation/"))

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()

download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>%
  mutate(movieId = as.numeric(levels(movieId))[movieId],
         title = as.character(title),
         genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)

edx <- movielens[-test_index, ]

temp <- movielens[test_index, ]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)

edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

####################################################################################################

# Define a function to calculate the RMSE

# This is how we will evaluate our algorithm

RMSE <- function(true_ratings, predicted_ratings) {
  
  sqrt(mean((true_ratings - predicted_ratings)^2))
  
}

# Check that all movie titles contain the movie release date by looking for
# the following regex pattern: \\s\\(\\d{4}\\)
# This looks for space, open parenthesis, four digits for the year, close parenthesis.
# The space at the beginning is not necessary, it just allows us to ensure that
# the structure of the title column is the following: TITLE (YEAR)

if (!all(str_detect(string = edx$title, pattern = '\\s\\(\\d{4}\\)'))) {
  
  warning("Check which movie titles do not follow the pattern: TITLE (YEAR)")
  
}

# Extract the movie release year from the title.
# Add it into a new column called 'release_year'.
# Add a new column called 'date' with the timestamp converted into datetime,
# rounded to the nearest week.

edx <- edx %>%
  mutate(
    # Create release_year column
    release_year = as.numeric(
      str_replace_all(
        string = str_extract(string = edx$title, pattern = '\\s\\(\\d{4}\\)'),
        pattern = "\\s\\(|\\)",
        replacement = ""
      ) 
    ),
    # Remove release year, i.e. (YEAR), from title column
    title = str_replace(
      string = edx$title,
      pattern = "\\s\\(\\d{4}\\)",
      replacement = ""
    ),
    # Create date column and round to nearest week
    date = round_date(as_datetime(timestamp), "week"),
    # Convert the 'genres' column into a factor
    genres = factor(genres),
    # Period between release_year and dataset release year
    rating_period = 2009 - release_year
  )

# Treat the validation set in the same way as the edx set

validation <- validation %>%
  mutate(
    # Create release_year column
    release_year = as.numeric(
      str_replace_all(
        string = str_extract(string = validation$title, pattern = '\\s\\(\\d{4}\\)'),
        pattern = "\\s\\(|\\)",
        replacement = ""
      ) 
    ),
    # Remove release year, i.e. (YEAR), from title column
    title = str_replace(
      string = validation$title,
      pattern = "\\s\\(\\d{4}\\)",
      replacement = ""
    ),
    # Create date column and round to nearest week
    date = round_date(as_datetime(timestamp), "week"),
    # Convert the 'genres' column into a factor
    genres = factor(genres),
    # Period between release_year and dataset release year
    rating_period = 2009 - release_year
  )

# Split the edx set into train_set and test_set

set.seed(1)

train_index <- createDataPartition(y = edx$rating, times = 1, p = 0.75, list = FALSE)

train_set <- edx[train_index, ]

temp <- edx[-train_index, ]

# Make sure userId and movieId in the test_set are also in train_set

test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test_set back into train_set

removed <- anti_join(temp, test_set) # generates a harmless message "Joining, by = c(..."

train_set <- rbind(train_set, removed)

rm(temp, removed)

#####

# Average rating.

####################################################################################################

# Just the average

mu_rating <- mean(train_set$rating) # avarage rating for all movies/users

rmse_results <- data.frame(model = "Just the average",
                           RMSE = RMSE(test_set$rating, mu_rating),
                           lambda = "-",
                           stringsAsFactors = FALSE)

#####

# The Movie and User Model.

####################################################################################################

# Movie differ in how they are rated

# Users differ in how they rate movies

# These are referred to as the movie and user effect respectively
# and can be accounted for to improve our model

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_rating)) # the movie model

user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_rating - b_i)) # the movie and user model

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu_rating + b_i + b_u) %>%
  .$pred

rmse_results <- bind_rows(rmse_results,
                          data.frame(model = "Movie and User Effects Model",
                                     RMSE = RMSE(predicted_ratings, test_set$rating),
                                     lambda = "-",
                                     stringsAsFactors = FALSE))

#####

# Regularized Movie and User Effect Model.

####################################################################################################

# Use a range of lambdas to choose the one that minimizes the RMSE

lambdas <- seq(4, 6, 0.2)

rmses <- sapply(lambdas, function(l) {
  
  print(paste0("Lambda = ", l))
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_rating) / (n() + l), n_i = n())
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_rating - b_i) / (n() + l))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_rating + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, test_set$rating))
  
})

# Make predictions using the optimum lambda

lambda <- lambdas[which.min(rmses)]

b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_rating) / (n() + lambda), n_i = n())

b_u <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_rating - b_i) / (n() + lambda))

predicted_ratings <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu_rating + b_i + b_u) %>%
  .$pred

rmse_results <- bind_rows(rmse_results,
                          data.frame(model = "Regularized Movie and User Effect Model",
                                     RMSE = RMSE(predicted_ratings, test_set$rating),
                                     lambda = as.character(lambda),
                                     stringsAsFactors = FALSE))

#####

# Regularized Movie, User, Genres, Rating Period and Time Effect Model.

####################################################################################################

# Use a range of lambdas to choose the one that minimizes the RMSE

lambdas <- seq(4, 6, 0.2)

rmses <- sapply(lambdas, function(l) {
  
  print(paste0("Lambda = ", l))
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_rating) / (n() + l), n_i = n())
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_rating - b_i) / (n() + l))
  
  b_g <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu_rating - b_i - b_u) / (n() + l))
  
  b_r <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    group_by(rating_period) %>%
    summarize(b_r = sum(rating - mu_rating - b_i - b_u - b_g) / (n() + l))
  
  b_t <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_r, by = "rating_period") %>%
    group_by(date) %>%
    summarize(b_t = sum(rating - mu_rating - b_i - b_u - b_g - b_r) / (n() + l))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_r, by = "rating_period") %>%
    left_join(b_t, by = "date") %>%
    mutate(pred = mu_rating + b_i + b_u + b_g + b_r + b_t) %>%
    .$pred
  
  return(RMSE(predicted_ratings, test_set$rating))
  
}) # sapply

qplot(x = lambdas, y = rmses) # plot the RMSE versus lambda

# Make predictions using the optimum lambda

lambda <- lambdas[which.min(rmses)]

b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_rating) / (n() + lambda), n_i = n())

b_u <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_rating - b_i) / (n() + lambda))

b_g <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_rating - b_i - b_u) / (n() + lambda))

b_r <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  group_by(rating_period) %>%
  summarize(b_r = sum(rating - mu_rating - b_i - b_u - b_g) / (n() + lambda))

b_t <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_r, by = "rating_period") %>%
  group_by(date) %>%
  summarize(b_t = sum(rating - mu_rating - b_i - b_u - b_g - b_r) / (n() + lambda))

predicted_ratings <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_r, by = "rating_period") %>%
  left_join(b_t, by = "date") %>%
  mutate(pred = mu_rating + b_i + b_u + b_g + b_r + b_t) %>%
  .$pred

rmse_results <- bind_rows(rmse_results,
                          data.frame(model = paste0("Regularized Movie, User, Genres, Rating Period ",
                                                    "and Date Effect Model"),
                                     RMSE = RMSE(predicted_ratings, test_set$rating),
                                     lambda = as.character(lambda),
                                     stringsAsFactors = FALSE))

# Calculate the improvement in RMSE between each model

model_improvement <- sapply(X = 1:(nrow(rmse_results) - 1), FUN = function(r) {
  
  current_row <- r
  
  next_row <- r + 1
  
  imp <- (rmse_results$RMSE[current_row] - rmse_results$RMSE[next_row]) * 100 /
    rmse_results$RMSE[current_row]
  
  return(round(imp, 4))
  
})

rmse_results <- rmse_results %>%
  mutate(improvement = c("-", model_improvement),
         RMSE = round(RMSE, 4)) %>%
  select(model, RMSE, improvement, lambda)

#####

# Validate model.

####################################################################################################

edx_rating <- mean(edx$rating) # avarage rating for all movies/users

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - edx_rating) / (n() + lambda), n_i = n())

b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - edx_rating - b_i) / (n() + lambda))

b_g <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - edx_rating - b_i - b_u) / (n() + lambda))

b_r <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  group_by(rating_period) %>%
  summarize(b_r = sum(rating - edx_rating - b_i - b_u - b_g) / (n() + lambda))

b_t <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_r, by = "rating_period") %>%
  group_by(date) %>%
  summarize(b_t = sum(rating - edx_rating - b_i - b_u - b_g - b_r) / (n() + lambda))

# Make predictions using the optimum lambda

predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_r, by = "rating_period") %>%
  left_join(b_t, by = "date") %>%
  mutate(pred = edx_rating + b_i + b_u + b_g + b_r + b_t) %>%
  .$pred

rmse_edx <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model = paste0("Validation set: ",
                                                    "Regularized Movie, User, Genres, Rating Period ",
                                                    "and Date Effect Model"),
                                     RMSE = round(rmse_edx, 4),
                                     improvement = "-",
                                     lambda = as.character(lambda),
                                     stringsAsFactors = FALSE))

print(paste0("Best model: ", rmse_results$model[which.min(rmse_results$RMSE)]))

print(paste0("Regularization lambda: ", lambda))

print(paste0("Minimum RMSE: ", round(min(rmse_results$RMSE), 4)))

#####
