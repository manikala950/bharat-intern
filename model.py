import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the Iris dataset
data = pd.read_csv('C:\\Users\\manik\\Desktop\\New folder (2)\\iris.csv')

# Split the data into features (X) and the target variable (y)
X = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Save the trained model to a file using pickle
with open('iris_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Later, you can load the model and use it for predictions
with open('iris_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


