# Data Mining Project 1: Regression Analysis on Concrete Data
This assignment is about applying data mining techniques on the UCI Machine Learning Repository dataset: Concrete Compressive Strength. The data showed 8 features with 1 response value (Concrete strength). 

The problem is finding the relation between these features contributing to the response value. I employed 8 univariate models (i.e. each model with one column as input feature) and 1 multivariate model (i.e. one model with 8 columns as input features).
By using gradient descent as the algorithm with Mean Squared Error (MSE) as a loss function, I calculated the R squared value according to the training set and testing set. By doing so, we can understand how a single feature contributes to the resulting outcome of concrete strength. 

In Bonus point_MAE_Ridge.ipynb, I tried different loss functions such as Mean Absolute Error (MAE) and Ridge Regression other than MSE. 

See Programming Assignment Description 1.pdf for detailed requirements.

## SP2023 Project 1.ipynb and Project 1 Final Report.pdf: 

These are the jupyter notebook consisting of the whole processing, and the final written report documenting the whole processing.

# Discussion and Main Findings: 

## 1. Did the same model perform equally to the training set and testing set?

In my case, yes. For example, for univariate models, it occasionally perform better using different loss functions. But overall, it performs pretty poorly on every type of loss function regardless of the training set or the testing set. On the other hand, the multivariate model generally performs well on MSE, MAE, and ridge regression. Especially with processed data, and overall the performance on the training set is similar to the testing set.

## 2. Did different models take longer to train or require different hyperparameters?

Yes, a model using raw data requires a smaller learning rate so that the weight would not overflow. Especially, I have encountered multiple times that the model with raw data tends to have the weight bouncing back and forth between two values, thus, I need to tune the learning rate into a smaller value to make it converge.

## 3. How did preprocessed data change the results?

Preprocessed data overall performs better than raw data, given that it handles the outliers and normalizes the data so that the model can read the value equally and easier.

## 4. What factor contributes the most?

The only feature that has the most influence is the cement. It constantly has the best performance on models. So, I would say the higher the value of cement has, the it is more likely to have stronger concrete strength.

## 5. MAE and Ridge

MAE overall did not change the result that much. I think it is because it has a similar loss calculation to MSE.
Ridge makes the multivariate model perform better. This may be because there may be high correlations between predictor variables, which can cause instability in the estimates of the regression coefficients. Ridge regression can help to reduce the impact of multicollinearity by adding a penalty term to the regression equation that shrinks the coefficients towards zero.




