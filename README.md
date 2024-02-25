# CSE151A_FINAL_PROJECT


Dataset contains 3313 observations and 9 features.
The goal of this project is to predict Laid off financial period. We add a new target column based on 'date' feature.

Numerical features data exploration:
- there are only 3 features are numerical. total_laid_off, percentage_laid_off, funds_raised
- total_laid_off contains 34 % missing values, percentage_laid_off contains 35 % missing values, funds_raised contains 11 % missing values
- Based on Plots for numerical features:
(1) For total_laid_off, the plot present Uniform distribution
(2) For percentage_laid_off, the plot present right skewed distribution
(3) For funds_raised, the plot present Uniform distribution
(4) The correlation between numerical features are not strong. Weak negative relationship between total_laid_off and percentage_laid_off; Weak positive relationship between total_laid_off and funds_raised;Weak negative relationship between funds_raised and percentage_laid_off; The highest relationship is between funds_raised and total_laid_off. The highest correlation between target 'quarters' is negative relationship between quarters and total_laid_off.
(5) There is no evident relationship between target and numerical features from pairplot

Categorical features data exploration:
- Only few missing values in location, industry, stage. However 'stage' contains a lot of Unkown value
- There are 6 categorical features: company, location, industry, stage, country, quarters(target)

Observation based on plots:
- Top 5 frequent industry in the datasets are:'Finance', 'Retail', 'Healthcare', 'Transportation', and other
- Top 5 frequent location in the datasets are: 'SF Bay Area', 'New York City', 'Boston', 'Bengaluru', 'Los Angeles',
       'Seattle'
- Top 5 frequent stage in the datasets are: 'Post-IPO', 'Unknown', 'Series B', 'Series C', 'Series D'
- Top 5 frequent country in the datasets are: 'United States', 'India', 'Canada', 'United Kingdom', 'Germany'
- Top 5 frequent company in the datasets are: 'Amazon', 'Uber', 'Spotify', 'Loft', 'Convoy'
- The most frequent quarter in the datasets is: Q2

Analysis based on heatmap:
- Relationship between industry and quarters. Layoffs in Finance industry occurs in quarter 2.
- Relationship between stage and quarters. Layoffs in stage post-ippo occurs in quarter 1.
- Relationship between locations and quarters. Layoffs in SF bay area occurs in quarter 2.
- Relationship between locations and quarters. Layoffs based on United States occurs in quarter 2.
- Relationship between company and quarters. Layoffs in Amazon occurs in quarter 2 and 4.



**Conclusion**
- Conclusion for Baseline model: Observing the difference in training error and test error, we believe the model has a sign of over fitting since the training error is much lower than the test error. In the fitting graph, the model may only successfully trace the pattern of training data, and is unable to predict the general pattern on unseen test data. The reason might be that the model is capturing too much noise in the training data and took such information into account of making predictions.
- Possible improvement of Baseline model: In order to improve the issue of over-fitting we decide to implement models with more layers. The baseline logistic regression fails because it's inability to capture the complex pattern in between categories. By adding layers into the model, we believe such an issue can be improved.

**Summary**:
- the dataset is balanced in term of target, the amount of each category of labels is similar.
- too many missing values in numerical features
- too many Unkown values in categorical feature 'stage'


**Next steps**:
- For the Next two models, we are thinking about exploring binary categorization and multi-class categorization with ANN/CNN> In which we are going to use activation functions 'softmax' and 'sigmoid' correspondingly, to see whether it is best to use  binary categorization and multi-class categorization on our numerical continuous but actually discrete target data 'quarters'.
