Methods section (this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, Model 3, (note models can be the same i.e. CNN but different versions of it if they are distinct enough). You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods


# METHOD
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

For this model, we follow similar steps. We first try the defualt model imported from the library, then optimize it with randomized search. For this project, we choose number of estimators and max depth as parameters for tuning. The number of iteration for searching is 10 and cross validation is 3. The scoring for searching is accuracy. After the hyperparameter tuning is done, we choose the best estimator for prediction.

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

The training process for this model is similar to Random Forest in the second experiment. And we use the same hyperparameter tuning method and values. After the tuning is done, we use the best estimator for prediction.

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

After experiments with different models, we do not see big difference in the results. So we decide to make change to the dataset with oversampling. We use RandomOverSampler to add more copies to minority classes.

```python
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
print('Original dataset shape %s' % Counter(y_train))
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train,y_train)
print('Resampled dataset shape %s' % Counter(y_res))
```

c. Ensemble Models (Voting)






# RESULTS
- loss plot
- accuracy

# DISCUSSION

# CONCLUSION
