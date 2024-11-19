**Appliances_Energy_prediction_ML_Model_Regression**
Appliance energy prediction using linear regression models Capstone-Project-On-Appliance_Energy_Prediction Introduction

In today's world, the use of energy is quickly expanding. We are experiencing a lack of energy as a result of increased energy consumption in some regions of the world, which is causing environmental damage. We are dealing with excessive energy consumption in home appliances in some places, so our main goal in this project is to analyze what factors are influencing the increasing energy consumption of home appliances, how we can reduce energy consumption of home appliances, and predict energy consumption of appliances using regression models. This project will analyze the Energy dataset using visualizations and graphs created with Python packages such as sklearn, matplotlib, Seaborn, and Eli5.

Abstract

Because of the transition towards the Internet of Everything, predicting the energy required by household appliances is a difficult study topic. The energy consumptions of appliances were forecasted in this work using an ML model-based strategy that employed the linear, lasso, ridge, decision tree, random forest, gradient boosting, xgb, adaboost, and lgbm regression algorithms. The current study has two goals: to maximize the prediction performance of the algorithms and to minimize the number of selected features. The proposed method was evaluated on an appliance energy prediction dataset taken from Reliable Prognosis' public dataset.

Dataset information

**Date time year- month-day hour:**minute:second

**Appliances- **energy use in Wh

**Lights- **energy use of light fixtures in the house in Wh

**T1- **Temperature in kitchen area, in Celsius

**RH_1- **Humidity in kitchen area, in %

**T2- **Temperature in living room area, in Celsius

**RH_2- **Humidity in living room area, in %

**T3- **Temperature in laundry room area

**RH_3- **Humidity in laundry room area, in %

**T4- **Temperature in office room, in Celsius

**RH_4- **Humidity in office room, in %

**T5- **Temperature in bathroom, in Celsius

**RH_5- **Humidity in bathroom, in %

**T6- **Temperature outside the building (north side), in Celsius

**RH_6- **Humidity outside the building (north side), in %

**T7- **Temperature in ironing room , in Celsius

**RH_7- **Humidity in ironing room, in %

**T8- **Temperature in teenager room 2, in Celsius

**RH_8- **Humidity in teenager room 2, in %

**T9- **Temperature in parents room, in Celsius

**RH_9- **Humidity in parents room, in %

**To- **Temperature outside (from Chievres weather station), in Celsius

**Pressure **(from Chievres weather station)- in mm Hg

**RH_out- **Humidity outside (from Chievres weather station), in %

**Wind speed **(from Chievres weather station)- in m/s

**Visibility **(from Chievres weather station)- in km

****Tdewpoint ****(from Chievres weather station)- Â°C

rv1- Random variable 1, nondimensional rv2- Random variable 2, nondimensional Where indicated, hourly data (then interpolated) from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis, rp5.ru. Permission was obtained from Reliable Prognosis for the distribution of the 4.5 months of weather data.

** Problem Statement**

For approximately 4.5 months, the data set is set to 10 minutes. A ZigBee wireless sensor network was used to monitor the house's temperature and humidity levels. Every wireless node reported the temperature and humidity levels every 3.3 minutes. The wireless data was then averaged over 10 minute periods. The energy data was recorded every 10 minutes using m-bus energy metres.

Weather from the nearest airport weather station (Chievres Airport, Belgium) was collected from a public data set from Reliable Prognosis (rp5.ru) and blended with the experimental data sets using the date and time columns. Two random variables were included in the data set to test the regression models and to filter out non-predictive features (parameters).

The problem statement is to create a machine learning model that can accurately forecast energy usage based on the supplied features. This might be valuable for building managers, energy firms, and policymakers who need to optimize energy consumption, cut costs, and minimize the environmental impact of energy usage.

Specifically, the model should be able to reliably anticipate energy usage based on the different elements that influence energy consumption, such as temperature, humidity, illumination, and time of day. This can assist building managers and energy firms in identifying patterns and trends in energy consumption and making informed energy decisions, such as altering HVAC settings, optimizing lighting, or introducing energy-efficient solutions. Policymakers can also use this data to create regulations and incentives that encourage energy efficiency and sustainability.

** Approaches**

This study seeks to forecast the energy usage of household appliances. With the emergence of smart homes and the growing demand for energy management, existing smart home systems can benefit from accurate predictions. If the energy usage of appliances can be forecasted for every potential condition, then device control can be optimized for energy savings as well. This is an example of Regression analysis, which is part of the Supervised Learning problem. The goal variable is appliance energy usage, and the characteristics are sensor data and weather data.

We will get some insights from data visualisation after retrieving the information from the date column and analyzing it so that we can determine the aspects that will influence the output, such as energy consumption by appliances, which is a big issue and one of the most pressing concerns for the Green Economy. For reproductive analysis, we will break this endeavor into five parts.

The first stage will be to analyze, clean up the dataset that was provided, and remove some extraneous columns.

In the next step we will draw some insights after fetching the other details from date column like hourly, daily and monthly energy consumption. After that we will do Feature selection and Engineering. In FE we will be using VIF and correlation both method for Removing Multicollinearity and drop Some unnecessary column.

Then we will split our data into train and test set after that we used some scaling techniques on train set and after that we prepare our data for feeding to our models for training and analyze the results of all models by comparing with each other. We will select best model among these and proceed with them further.

Interpretation of the Results:

MSE (Mean Squared Error): A measure of the average squared difference between actual and predicted values. Lower MSE values indicate a better fit, meaning the model's predictions are closer to the actual values.

Lower MSE is generally better because it means the model is making fewer large errors.

RMSE (Root Mean Squared Error): The square root of MSE. It provides a metric that's in the same units as the target variable, making it more interpretable. Lower RMSE is better.

Lower RMSE means the predictions are closer to actual values, with fewer large errors.

MAE (Mean Absolute Error): The average of the absolute errors between actual and predicted values. A lower MAE indicates a model with fewer large errors.

Lower MAE means the model's predictions are closer to the actual values on average.

R² (R-squared): Indicates how much of the variance in the target variable is explained by the model. R² ranges from 0 to 1, where 1 means the model explains all the variance, and 0 means it explains none. A higher R² score indicates better model performance.

Higher R² means the model does a better job of explaining the variation in the data.
Now, let’s break down the results:
1. Linear Regression:

    MSE: 0.247 (relatively high) 
    RMSE: 0.497 (indicates average error of about 0.497 units in the target variable)
    MAE: 0.364 (on average, the model is off by 0.364 units)
    R² Score: 0.280 (This indicates that only about 28% of the variance in the target variable is explained by the model. It's relatively low, suggesting that the model isn't performing very well.)

Conclusion: Linear regression doesn't seem to fit the data very well. The R² score is quite low, meaning the model isn’t capturing the underlying patterns effectively.

2. Decision Tree Regressor:

    MSE: 0.157 (lower than Linear Regression, which is a positive sign)
    RMSE: 0.396 (lower than Linear Regression's RMSE, suggesting the model is making fewer large errors)
    MAE: 0.244 (lower than Linear Regression, meaning the model's predictions are more accurate on average)
    R² Score: 0.542 (about 54% of the variance is explained by the model, which is a significant improvement over Linear Regression)

Conclusion: The Decision Tree Regressor performs better than Linear Regression, with a lower error and a higher R² score, indicating it explains more variance in the target variable.

3. Random Forest Regressor:

    MSE: 0.089 (even lower than Decision Tree, indicating very good performance)
    RMSE: 0.298 (lower than both Decision Tree and Linear Regression, indicating fewer errors)
    MAE: 0.207 (again, lower than the other models, indicating more accurate predictions)
    R² Score: 0.741 (about 74% of the variance is explained by the model, which is quite good)

Conclusion: Random Forest is performing the best out of all the models so far. It has the lowest error metrics (MSE, RMSE, MAE) and the highest R² score, meaning it does a good job of explaining the data and making accurate predictions.

4. Gradient Boosting Regressor:
    MSE: 0.184 (higher than Random Forest, but lower than Decision Tree)
    RMSE: 0.429 (higher than Random Forest, suggesting the model is making larger errors)
    MAE: 0.308 (also higher than Random Forest, meaning the model’s average prediction error is larger)
    R² Score: 0.464 (about 46% of the variance is explained by the model, which is lower than Random Forest)
    
Conclusion: While Gradient Boosting performs better than Linear Regression and Decision Tree, it still lags behind Random Forest in terms of both error metrics and R² score.

5. Support Vector Regressor (SVR):

    MSE: 0.284 (relatively high, similar to Linear Regression)
    RMSE: 0.533 (the highest RMSE, indicating the model is making large errors)
    MAE: 0.374 (the highest MAE, meaning predictions are, on average, less accurate)
    R² Score: 0.171 (only about 17% of the variance is explained, which is very low)
    
Conclusion: The Support Vector Regressor performs the worst among the models, with the highest errors and the lowest R² score.

### Final Model Evaluation:

Best Model: 
    Random Forest Regressor is clearly the best model here with the lowest error metrics (MSE, RMSE, MAE) and the highest R² score (0.741), meaning it explains the most variance in the data.

Second Best:
    Decision Tree Regressor performs better than other models except Random Forest. It is a good option for this dataset.

Worst Model:
    Support Vector Regressor performs poorly across all metrics, with the highest error values and the lowest R² score.
