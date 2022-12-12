# Wine-Quality-Prediction

There are 1599 samples of red wine. Each wine sample (row) has the following characteristics (columns):

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality (score between 0 and 10)

The study context is to try different predictive algorithms on the data and examine the results.

### Approach 
For the initial modelling I've used LazyPredict as it helps build a lot of basic models without much code and helps understand which models works better without any parameter tuning.
After that we can apply scaling and other feature engineering techniques to increase the accuracy of our top model which we can easily get from the lazypredict library.

Now after scaling I used PCA to reduce the dimensionality or the numbers of features that we are taking for the model.

After that we can also use GridSearchCV or RandomizedSearchCV for increasing the accuracy and apply hyperparameter tuning. 

We can now save the model using pickle library.

### Creating Flask Application
For the model I've created a basic flask application. The Static folder contains the CSS sheet and the templated folder contains the index.html sheet which is nothing but our home page.
Few customizations were done and voila, we have our own flask application which can be hosted on various different platforms.

