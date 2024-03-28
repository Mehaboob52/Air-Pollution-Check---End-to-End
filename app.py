from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# Load trained model, scaler, and label encoder from the air folder
model_path = "air/best_rf_model.pkl"
scaler_path = "air/scaler.pkl"
encoder_path = "air/region_encoder.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

input_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'Temperature', 'Humidity', 'Wind_Speed', 'Region']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    input_data = request.form.to_dict()
    
    # Extract the region value from the input data
    region = input_data.pop('Region', None)
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])
    
    # Encode the 'Region' value using the label encoder
    region_encoded = encoder.transform([region])[0]
    
    # Add the encoded 'Region' value to the input DataFrame
    input_df['Region'] = region_encoded
    
    # Perform one-hot encoding for 'Region'
    input_df = pd.get_dummies(input_df, columns=['Region'])
    
    # Reorder columns to match the model's expected input order
    input_df = input_df.reindex(columns=input_features)
    
    # Preprocess input data
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    # Return prediction as JSON response
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
