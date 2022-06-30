# MovieLens Capstone Project 
# Completed for the Harvard Data Science Professional Certificate 


### Repo Contents

This repository includes the following items: 
- R Script 
- An Rmarkdown version 
- Knitted PDF version of .rmd file


### General Info

The MovieLens recommendation site was launched in 1997 by GroupLens Research (which is part of the University of Minnesota). Today, the MovieLens database is widely used for research and education purposes. In total, there are approximately 11 million ratings and 8,500 movies. Each movie is rated by a user on a scale from 1/2 star up to 5 stars.

The goal of this project is to train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set. The key steps that are performed include:

-   Perform exploratory analysis on the data set in order to identify valuable variables

-   Generate a naive model to define a baseline RMSE and reference point for additional methods

-   Generate linear models using average movie ratings (movie effects) and average user ratings (user effects)

-   Utilize matrix factorization to achieve an RMSE below the desired threshold

-   Present results and report conclusions
  
### Results

Linear models did not provide sufficiently low RMSEs, however matrix factorization did achieve the desrired RMSE threshold of < 0.86490. The final test set RMSE was 0.83461 and the final validation set RMSE was 0.83396. Results are presented by method in the summary table below. 

![Final Table](https://github.com/edenaxe/DS_Capstone/blob/main/Images/Final_Table.PNG)

### Additional Notes

***Certificate info can be found [here](https://pll.harvard.edu/series/professional-certificate-data-science)
