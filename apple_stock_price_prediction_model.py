from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics
import numpy as np

# Load dataset
df = pd.read_csv('apple_stock.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[['Open', 'High', 'Low']] = imputer.fit_transform(df[['Open', 'High', 'Low']])

# Handle outliers
Q1 = df[['Open', 'High', 'Low']].quantile(0.25)
Q3 = df[['Open', 'High', 'Low']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['Open', 'High', 'Low']] < (Q1 - 1.5 * IQR)) | (df[['Open', 'High', 'Low']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# features & target
features = df[['Open', 'High', 'Low']]
target = df['Close']

# Normalize features
scaler = RobustScaler()
features = scaler.fit_transform(features)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Create pipeline
model = make_pipeline(StandardScaler(), LinearRegression())

# Training the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print coefficients
print('Coefficients: \n', model.named_steps['linearregression'].coef_)

# Print mean squared error
print('Mean squared error: %.2f' % metrics.mean_squared_error(y_test, y_pred))

# Print coefficient of determination
print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))

# Visualizing
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

