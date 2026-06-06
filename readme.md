# House-Price-Prediction

Created two models for the competition: [House Prices - Advanced Regression Techniques Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) on Kaggle. 

The first model is Linear Regression with Gradient Descent and achieved an RMSE score of 0.12875 on the test set. The predictions of this model is stored in the linear_regression_prediction.csv file.

The second model was a Stacking Model. It combined the predictions from a Random Forest Model, XGBoost Model, Ridge Regressor, Gradient Boosting Regressor, LightGBM Regressor and CatBoost Regressor, and used Ridge Regressor as the Meta-Model which combines the predictions in the most optimal way. 

The stacking model generates different predictions everytime. I ran the model twice and stored the predictions in stacking_prediction.csv and stacking_prediction2.csv, which achieved an RMSE score of 0.12172 and 0.12167 respectively. 

Both models use feature engineering, however stacking does a bit more, as some of the feature engineering was increasing the RMSE score for the linear regression model. 
