import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\OMOPC58\Downloads\dviya\copy.csv")

# Drop the 'Link' column
df = df.drop(columns=['Link'])

# Split the data into features (X) and target (y)
X = df[['area']]  # Feature (area)
y_volume = df['Volume (ml)']  # Target for volume prediction

# Split into training and testing sets
X_train, X_test, y_volume_train, y_volume_test = train_test_split(X, y_volume, test_size=0.2, random_state=42)

# Train the Random Forest Regressor for volume prediction
volume_model = RandomForestRegressor(n_estimators=100, random_state=42)
volume_model.fit(X_train, y_volume_train)

# Make predictions on the test set
y_volume_pred = volume_model.predict(X_test)

# Calculate confidence levels (standard deviation for regressors)
volume_confidence = np.std([tree.predict(X_test) for tree in volume_model.estimators_], axis=0)

# Evaluate the model
volume_mae = mean_squared_error(y_volume_test, y_volume_pred, squared=False)  # Root Mean Squared Error
print(f'Volume MAE: {volume_mae:.2f}')

# Display predictions with confidence levels
results_volume = X_test.copy()
results_volume['Predicted Volume'] = y_volume_pred
results_volume['Volume Confidence'] = volume_confidence
print(results_volume.head())

# Feature Importance for Volume Model
importances = volume_model.feature_importances_
print("Feature importances for volume model:", importances)

# Optional: Visualization of Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances, align='center')
plt.xticks(range(len(importances)), X.columns, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances for Volume Model')
plt.show()

# Optional: Visualization of Predictions
plt.scatter(y_volume_test, y_volume_pred)
plt.xlabel('Actual Volume')
plt.ylabel('Predicted Volume')
plt.title('Actual vs. Predicted Volume')
plt.plot([min(y_volume_test), max(y_volume_test)], [min(y_volume_test), max(y_volume_test)], color='red')
plt.show()
