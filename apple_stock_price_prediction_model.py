from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics

# Load dataset
df = pd.read_csv('apple_stock.csv')

# features & target
features = df[['Open', 'High', 'Low']]
target = df['Close']

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

