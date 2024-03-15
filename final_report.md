
# INTRODUCTION

Layoffs have become a sensitive but recurring topic after covid-19. Especially in the field of technology, undergrads like us are frequently receiving information about our ideal future working companies going through massive layoff. Layoff shows it’s tendency of growing after covid-19 which caused financial shortage and lack of fundings. The emergence of Chat GPT and other AI tools give companies more motivation to shift from human resources to investments in AI. Also, critics claim 2023’s massive layoff generated rewarding rewards from wall street such as the growth of stock price and thus leads to the loop of layoff increasement. To further understand the correlation behind layoff and various aforementioned factors, we carefully selected a dataset consist of companies’ information with detailed layoff record from 2019 to 2023, aiming to provide insight for undergrads and individuals who concerns what factors causes a company to consider layoff and the specific time of their action.




# METHOD
Methods section (this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, Model 3, (note models can be the same i.e. CNN but different versions of it if they are distinct enough). You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods

## Exploration results

Dataset contains 3313 observations and 9 features.
The goal of this project is to predict Laid off financial period. We add a new target column based on 'date' feature.

*Numerical features data exploration*:
- there are only 3 features are numerical. total_laid_off, percentage_laid_off, funds_raised
- total_laid_off contains 34 % missing values, percentage_laid_off contains 35 % missing values, funds_raised contains 11 % missing values
- Based on Plots for numerical features:
(1) For total_laid_off, the plot present Uniform distribution
(2) For percentage_laid_off, the plot present right skewed distribution
(3) For funds_raised, the plot present Uniform distribution
(4) The correlation between numerical features are not strong. Weak negative relationship between total_laid_off and percentage_laid_off; Weak positive relationship between total_laid_off and funds_raised;Weak negative relationship between funds_raised and percentage_laid_off; The highest relationship is between funds_raised and total_laid_off. The highest correlation between target 'quarters' is negative relationship between quarters and total_laid_off.
(5) There is no evident relationship between target and numerical features from pairplot

*Categorical features data exploration*:
- Only few missing values in location, industry, stage. However 'stage' contains a lot of Unkown value
- There are 6 categorical features: company, location, industry, stage, country, quarters(target)

*Observation based on plots*:
- Top 5 frequent industry in the datasets are:'Finance', 'Retail', 'Healthcare', 'Transportation', and other
- Top 5 frequent location in the datasets are: 'SF Bay Area', 'New York City', 'Boston', 'Bengaluru', 'Los Angeles',
       'Seattle'
- Top 5 frequent stage in the datasets are: 'Post-IPO', 'Unknown', 'Series B', 'Series C', 'Series D'
- Top 5 frequent country in the datasets are: 'United States', 'India', 'Canada', 'United Kingdom', 'Germany'
- Top 5 frequent company in the datasets are: 'Amazon', 'Uber', 'Spotify', 'Loft', 'Convoy'
- The most frequent quarter in the datasets is: Q2

*Analysis based on heatmap*:
- Relationship between industry and quarters. Layoffs in Finance industry occurs in quarter 2.
- Relationship between stage and quarters. Layoffs in stage post-ippo occurs in quarter 1.
- Relationship between locations and quarters. Layoffs in SF bay area occurs in quarter 2.
- Relationship between locations and quarters. Layoffs based on United States occurs in quarter 2.
- Relationship between company and quarters. Layoffs in Amazon occurs in quarter 2 and 4.

## Preprocessing steps
- Handle missing values: use imputation to replace missing numerical values with the mean of their respective columns. Since categorical values only have 9 missing values. We decide jsut drop the rows.
- Rescale data: use MinMaxScaler to normalize the numerical features of the dataset to a range between 0 and 1.
- Transform categorical features: use one hot encoder.

## Model 1
a. Logistic Regression

```python
model_lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
```

b. Sequential Model

- For this neural network model, one extra step before training the model is to preprocess the label with one hot encoder. We also try to optimize the model performance with different activation functions, number of nodes, layers, loss function and early stopping.

```python
def build_model():
    sequential_model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')
    ])
    sequential_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return sequential_model

sequential_model = KerasClassifier(build_fn=build_model, epochs=100, batch_size=32, verbose=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = sequential_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
```

## Model 2
a. SVM
- For this model, we need to have an extra step to transform the dataset. Previously, we transformed the label with one hot encoder for training neural network. But this format is not suitable for training SVM. And the goal of this project is to multi class classfication, we decided to use one versu rest. We experiment two basic kernel functions: linear and rbf. After we try these two basic models, we also perform hyper tuning with randomized search.
```python
svm_linear = SVC(kernel='linear', decision_function_shape='ovr')
svm_rbf = SVC(kernel='rbf', decision_function_shape='ovr')

param_dist = {'C': [0.1, 1, 10],
              'gamma': [1, 0.1, 0.01],
              'kernel': ['linear','rbf']}

svm = SVC(decision_function_shape='ovr',random_state=42, probability=True)
random_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1, scoring='accuracy')
random_search.fit(X_train, y_train)
best_svm = random_search.best_estimator_

```

b. Random Forest

- For this model, we follow similar steps. We first try the defualt model imported from the library, then optimize it with randomized search. For this project, we choose number of estimators and max depth as parameters for tuning. The number of iteration for searching is 10 and cross validation is 3. The scoring for searching is accuracy. After the hyperparameter tuning is done, we choose the best estimator for prediction.

```python

param_dist = {
    'n_estimators': [10, 20,30,40],
    'max_depth': [None, 10, 20,30],
}

rf_clf = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(rf_clf, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
rnd_search.fit(X_train, y_train)
best_params = rnd_search.best_params_
best_score = rnd_search.best_score_
rf_best = RandomForestClassifier(**best_params, random_state=42)
```

## Model 3
a. Gradient Boosting

- The training process for this model is similar to Random Forest in the second experiment. And we use the same hyperparameter tuning method and values. After the tuning is done, we use the best estimator for prediction.

```python
param_dist = {
    'n_estimators': [10, 20,30,40],
    'max_depth': [None, 10, 20,30]
}
gb_clf = GradientBoostingClassifier(random_state=42)

random_search = RandomizedSearchCV(estimator=gb_clf, param_distributions=param_dist, cv=3, n_jobs=-1, scoring='accuracy')

random_search.fit(X_train, y_train)

gb_best = random_search.best_estimator_
```

b. Oversampling

- After experiments with different models, we do not see big difference in the results. So we decide to make change to the dataset with oversampling. We use RandomOverSampler to add more copies to minority classes.

```python
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
print('Original dataset shape %s' % Counter(y_train))
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train,y_train)
print('Resampled dataset shape %s' % Counter(y_res))
```

c. Ensemble Models (Voting)

- After we try oversampling the dataset, we decide to apply the ensemble techniques voting to improve the performance. We use voting classifier to train on multiple models and predict the target based on the average of probability given to that class. We do not include the sequential model in this case because of the incompability tensorslow, and based on previous experiment sequential model shows the lowest accuracy, so we believe that excluding sequential model in this experiment would not make big difference.


```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('lr', model_lg), ('svm', best_svm),
        ('dtrf', rf_best), ('gb', gb_clf)    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
```






# RESULTS
## Prepocessing Results
a. handle missing values

![alt text](image-14.png)

b. Rescale data

![alt text](image-15.png)

c. Transform categorical features

![alt text](image-16.png)

## Model 1 results
a. Logistic Regression
![alt text](image-3.png)

b. sequential Model
![alt text](image-4.png)

## Model 2 results
a. SVM
![alt text](image-12.png)
![alt text](image-13.png)

b. Random Forest
![alt text](image-5.png)
![alt text](image-6.png)

## model 3 results
a. Gradient Boosting
![alt text](image-7.png)
![alt text](image-8.png)

b. Oversampling
![alt text](image-10.png)

c. Ensemble Voting Classfier
![alt text](image-11.png)


Result table
| Models |  #Training Accuracy | #Test Accuracy  |
| ------- | --- | --- |
| Logistic Regression | 0.804 | 0.301 |
| Sequential Model | 0.368 | 0.308 |
| Best SVM | 0.393 | 0.319 |
| Best Random Forest | 0.539 | 0.308 |
| Best Gradient Boosting | 0.741 | 0.319 |
| Voting Classifier | 0.645 | 0.329 |

# DISCUSSION

Discussion section: This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!
## Data exploration
- In this section we use df.describe() function to have a general idea about the values in the dataframe. Then we do data visualization with histplot for the numerical features and categorical features. We use visualization because it is helpful to see the distribution of numerical data. Then we use correlation plot and correlation matrix to see any potential relationship between features, because this helps to improve the model performance in the later steps of this project, the last step we do in data exploration is checking the frequency of categorical data, this is important because we want to see whether there is any imbalanced informaiton in the features data and the label.

## Prepocessing Results
a. handle missing values
- we notice there are missing values in the dataset, and missing values can lead to possible low performance in model training, so we decide to handle the missing values by using the statistic mean of each numerical data. Since numerical data contains a big amount of missing values,total_laid_off contains 34 % missing values
percentage_laid_off contains 35 % missing values
funds_raised contains 11 % missing values. We do not want to simply remove the corresponding rows, because it will cause information loss. We choose to impute with mean instead of 0, because we think 0 cannot be a representative value of the dataset. However, filling with mean also has disadvantage, which does not reflect the real situation if we want to apply the result of the project to real world scenario.


b. Rescale data
- After we finished filling the missing values, we observe the data values again and notice the all the numerical features have different minimum and maximum range. So we rescale all the elements with normalizaiton and make them lie between 0 and 1.

c. Transform categorical features
- The dataset contains a mix of numerical and categorical information. Decision trees can work with input without transformation, however some models like logisctic regression cannot process with categorical data directly. So we decide to use one hot encoder to convert strings into numerical values. However, the drawback of this is it generates a big quantity of new variables and causes possible multicollinearity among variables and lower the model's accuracy and slows the training process and adds more complexity.


## Model 1 results
a. Logistic Regression


b. sequential Model


## Model 2 results
a. SVM

b. Random Forest


## model 3 results
a. Gradient Boosting


b. Oversampling


c. Ensemble Voting Classfier



# CONCLUSION
