# Movie Recommendation System

## Overview

This project is part of the final section of the [Professional Certificate in Data Science by HarvardX](https://online-learning.harvard.edu/series/professional-certificate-data-science) offer by [edX](https://www.edx.org). The aim of the project was to develop a movie recommendation system. The data for this study were obtained from the [MovieLens 10M Dataset](https://grouplens.org/datasets/movielens/10m/).

The linear regression model developed was able to predict movie ratings for each user with reasonable accuracy.

More information about the methodology used and the results obtained can be found in the *movielens.pdf* report while *movielens_analysis.R* contains the code for testing, developing and evaluating the model.

## Repository contents

* **movielens.pdf**
    * The final PDF report knitted from an R Markdown file (see movielens.Rmd)

* **movielens.Rmd**
    * An R Markdown document including the code to generate all the plots, tables and results reported in movielens.pdf
    * Detailed instructions about how to reproduce the analysis are included

* **movielens_analysis.R**
    * An R script that will build the linear regression model, fit it and make predictions on a validation set
    * At the end, the model with the minimum Root Mean Square Error (RMSE) is returned

---
