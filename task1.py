import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'square_feet': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [3, 4, 3, 5, 4],
    'bathrooms': [2, 2, 3, 4, 3],
    'price': [300000, 400000, 450000, 600000, 650000]
}

df = pd.DataFrame(data)

X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

for i in range(len(y_test)):
    print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Fixed prediction for a new house
new_house = pd.DataFrame([[2500, 4, 3]], columns=['square_feet', 'bedrooms', 'bathrooms'])
predicted_price = model.predict(new_house)
print(f"Predicted price for new house: ${predicted_price[0]:,.2f}")
