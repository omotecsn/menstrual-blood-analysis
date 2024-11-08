import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\OMOPC58\Downloads\dviya\copy.csv")

# Mapping viscosity values to labels
viscosity_map = {
    '5g/100ml': 1,
    '7.5g/100ml': 2,
    '10g/100ml': 3
}
df['Viscosity_Label'] = df['Viscosity'].map(viscosity_map)

# Drop the 'Link' column
df = df.drop(columns=['Link'])

# Split the data into features (X) and target (y)
X = df[['area']]  # Feature (area)
y_viscosity = df['Viscosity_Label']  # Target for viscosity label prediction

# Split into training and testing sets
X_train, X_test, y_viscosity_train, y_viscosity_test = train_test_split(X, y_viscosity, test_size=0.2, random_state=42)

# Train the Random Forest Classifier for viscosity prediction
viscosity_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
viscosity_model.fit(X_train, y_viscosity_train)

# Make predictions on the test set
y_viscosity_pred = viscosity_model.predict(X_test)

# Calculate confidence levels (probabilities for classifiers)
viscosity_confidence = np.max(viscosity_model.predict_proba(X_test), axis=1)

# Evaluate the model
viscosity_accuracy = accuracy_score(y_viscosity_test, y_viscosity_pred)
print(f'Viscosity Accuracy: {viscosity_accuracy:.2f}')

# Display predictions with confidence levels
results_viscosity = X_test.copy()
results_viscosity['Predicted Viscosity'] = y_viscosity_pred
results_viscosity['Viscosity Confidence'] = viscosity_confidence
print(results_viscosity.head())

# Optional: Confusion Matrix for Viscosity Prediction
cm = confusion_matrix(y_viscosity_test, y_viscosity_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for Viscosity Prediction')
plt.show()

# Hyperparameter Tuning for Viscosity Model
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1_macro', cv=5)
grid_search.fit(X_train, y_viscosity_train)
best_model = grid_search.best_estimator_

# Evaluate the best model found by Grid Search
best_viscosity_pred = best_model.predict(X_test)
best_viscosity_accuracy = accuracy_score(y_viscosity_test, best_viscosity_pred)
print(f'Best Model Accuracy: {best_viscosity_accuracy:.2f}')
