# Predictive Model for Apple Stock Prices

This repository contains a Python script for building a predictive model using linear regression to predict the closing price of Apple's stock based on the opening, high, and low prices.

This project is done as a part of "CodeAlpha Internship (Data Science)" Task_3
STUDENT ID: CA/DS/12451 
Name: A.N. AL IMAN

## Dataset
The dataset used in this project is `apple_stock.csv`, which contains historical stock prices for Apple Inc.

## Features
The features used for prediction are:
- 'Open': The opening price of the stock for the day
- 'High': The highest price of the stock for the day
- 'Low': The lowest price of the stock for the day

## Target
The target variable is 'Close', which is the closing price of the stock for the day.

## Model
The model is a pipeline that first scales the features using `StandardScaler` and then applies `LinearRegression`.

## Evaluation Metrics
The model is evaluated using the following metrics:
- Mean Squared Error (MSE)
- Coefficient of Determination (R^2 Score)

## Visualization
A scatter plot is generated to visualize the actual vs predicted values.

## Usage
To run the script, use the following command:

```python3 apple_stock_price_prediction.py```

## Requirements
The script requires the following Python libraries:

pandas
sklearn
seaborn
matplotlib

## License
This project is licensed under the MIT License.




