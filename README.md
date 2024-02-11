# CSE151A_FINAL_PROJECT


Dataset contains 3313 observations and 9 features.

Numerical features data exploration:
- there are only 3 features are numerical. total_laid_off, percentage_laid_off, funds_raised
- total_laid_off contains 34 % missing values, percentage_laid_off contains 35 % missing values, funds_raised contains 11 % missing values
- Based on Plots for numerical features:
(1) For total_laid_off, the plot present Uniform distribution
(2) For percentage_laid_off, the plot present right skewed distribution
(3) For funds_raised, the plot present Uniform distribution
(4) The correlation between numerical features are not strong. Weak negative relationship between total_laid_off and percentage_laid_off; Weak positive relationship between total_laid_off and funds_raised;Weak negative relationship between funds_raised and percentage_laid_off; The highest relationship is between funds_raised and percentage_laid_off.


Categorical features data exploration:



Summary:
- unbalanced datasets
- too many missing values in numerical features


Next steps:
- Handle missing values by (1)fill Null values with mean (2) drop rows contain NaN and add more datapoints from backup datasets.
- Normalize numerical features
- Encode categorical features
- Process the new Target column based on feature 'date'
