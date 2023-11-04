# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load your dataset (replace 'data.csv' with your dataset file)
data = pd.read_csv('C:\\Users\\manik\\Desktop\\New folder (3)\\house.csv')

# Feature selection
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(11, 11))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
threshold = 0.4
selected_features = corrmat.columns[corrmat.abs().mean() > threshold]
print("Selected features are:", selected_features)

# Split the data into training and testing sets
X = data[selected_features]
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

# Later, you can load the model and use it for predictions
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


