import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ensure that the air directory exists
if not os.path.exists("air"):
    os.makedirs("air")

# Load the data
data = pd.read_csv("D:\\ml datasets\\air.csv")
data = data.drop(columns=["Date"])

# Drop rows with missing values
data.dropna(inplace=True)

# Separate features and target variable
x = data.drop(columns=["Target_Variable"])  
y = data["Target_Variable"] 

# Label encode the 'Region' column
encoder = LabelEncoder()
x['Region'] = encoder.fit_transform(x['Region'])

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    # Add other hyperparameters as needed
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
best_rf_model = RandomForestRegressor(**best_params)
best_rf_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = best_rf_model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)

# Save the trained model, scaler, and label encoder into the air folder
model_path = "air/best_rf_model.pkl"
scaler_path = "air/scaler.pkl"
encoder_path = "air/region_encoder.pkl"

joblib.dump(best_rf_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)
