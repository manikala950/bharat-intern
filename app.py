# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

from flask import Flask, render_template, request
import pymongo

app = Flask(__name__)

# Load your dataset (replace 'data.csv' with your dataset file)
data = pd.read_csv('C:\\Users\\manik\\Desktop\\New folder (3)\\house.csv')

# Feature selection
corrmat = data.corr()
top_corr_features = corrmat.index
threshold = 0.1
selected_features = corrmat.columns[corrmat.abs().mean() > threshold]
print("Selected features are:", selected_features)

# Split the data into training and testing sets
X = data[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Set up the MongoDB connection
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['HOME']  # Replace 'your_database_name' with your actual database name
collection = db["PREDICTION"]
# Define a route for rendering the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for handling the form submission and storing data in MongoDB
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    income = float(request.form['income'])
    house_age = float(request.form['house_age'])
    num_rooms = float(request.form['num_rooms'])
    num_bedrooms = float(request.form['num_bedrooms'])
    population = float(request.form['population'])

    # Perform your prediction using the machine learning model
    input_data = [[income, house_age, num_rooms, num_bedrooms, population]]
    predicted_price = model.predict(input_data)[0]

    # Save the prediction data to MongoDB
    prediction_data = {
        'income': income,
        'house_age': house_age,
        'num_rooms': num_rooms,
        'num_bedrooms': num_bedrooms,
        'population': population,
        'predicted_price': predicted_price
    }

    
    collection.insert_one(prediction_data)

    # Render the result page with the prediction result
    return render_template('result.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
