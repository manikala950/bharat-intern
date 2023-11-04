import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
import pickle

# Load the Iris dataset
data = pd.read_csv('C:\\Users\\manik\\Desktop\\New folder (2)\\iris.csv')

# Split the data into features (X) and the target variable (y)
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('iris_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Load the trained model
with open('iris_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def index():
    return "Iris Flower Species Prediction"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sepal_length = data['sepal_length']
        sepal_width = data['sepal_width']
        petal_length = data['petal_length']
        petal_width = data['petal_width']

        # Make a prediction using the loaded model
        prediction = loaded_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main':
    app.run(debug=True)
