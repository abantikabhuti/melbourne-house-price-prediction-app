########### 1. Import Libraries ###########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

########### 2. Load Dataset ###########

df = pd.read_csv("melb_data.csv")
df.head()

########### 3. Data Preprocessing ###########

# Check column names
print('Available columns:',df.columns.to_list())

# Drop irrelevant columns
df = df.drop(['Method','Regionname'],axis=1)
print('Available columns:',df.columns.to_list())

# Encode categorical variable 'Type'
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# Define feature and target set (only existing columns)
features = ['Rooms','Type','Postcode','Distance','Propertycount']
target = ['Price']

# Drop missing values
df = df.dropna()

# Select features and target
X = df[features]
y = df[target]

########### 4. Exploratory Data Analysis ###########

# Feature Correlation
plt.figure(figsize=(8,8))
sns.heatmap(X.corr(numeric_only=True), annot=True, cmap='rocket')
plt.title('Feature Correlation')
plt.show()

# Distribution of House Prices
sns.histplot(y, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

########### 5. Train/Test Split ###########

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########### 6. Model Training with XGBoost ###########

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42
)

model.fit(X_train, y_train)

########### 7. Model Evaluation ###########

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

########### 8. Visualize Predictions ###########

print(type(y_pred))
print(type(y_test))

# Converting y_test to Series for plotting
y_test_series = y_test['Price']

plt.scatter(y_test_series, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('XGBoost: Predicted vs Actual Price')
plt.plot([min(y_test_series), max(y_test_series)], [min(y_test_series), max(y_test_series)], color='red')
plt.show()

########### 9. Feature Importance ###########

from xgboost import plot_importance

plot_importance(model)
plt.title('Feature Importance: XGBoost')
plt.show()

########### 10. Make Predictions ###########

# Example input: [Rooms, Type, Distance, Propertycount, Postcode]
# Type must be encoded (e.g. h=0, u=1, t=2)
sample_input = np.array([[3,0,3000,10.2,1000]])
predicted_price = model.predict(sample_input)
price = round(predicted_price[0], 0)    # round off to nearest integer
print(f'Predicted Price: ${price}')

########### 11. Save Model ###########

import joblib

joblib.dump(model, 'xgboost_house_price_model.pkl')
print("✅ Model saved as 'xgboost_house_price_model.pkl'")
