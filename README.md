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
- Only few missing values in location, industry, stage. However 'stage' contains a lot of Unkown value
- There are 6 categorical features: company, location, industry, stage, country, date
- Top 5 frequent industry in the datasets are:'Finance', 'Retail', 'Healthcare', 'Transportation', and other
- Top 5 frequent location in the datasets are: 'SF Bay Area', 'New York City', 'Boston', 'Bengaluru', 'Los Angeles',
       'Seattle'
- Top 5 frequent stage in the datasets are: 'Post-IPO', 'Unknown', 'Series B', 'Series C', 'Series D'
- Top 5 frequent country in the datasets are: 'United States', 'India', 'Canada', 'United Kingdom', 'Germany'
- Top 5 frequent company in the datasets are: 'Amazon', 'Uber', 'Spotify', 'Loft', 'Convoy'



Summary:
- unbalanced datasets
- too many missing values in numerical features
- too many Unkown values in categorical feature 'stage'


Next steps:
- Process the new Target column based on feature 'date'
- Handle missing values by (1)fill Null values with mean (2) drop rows contain NaN and add more datapoints from backup datasets.
- Normalize numerical features
- Encode categorical features
