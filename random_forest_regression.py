import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
a = pd.read_csv("synthetic_scatter_data.csv")
x = a.drop("size_um", axis=1)  # features
y = a["size_um"]               # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create model
randomforest = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
print(X_train.shape)

# Train
randomforest.fit(X_train, y_train)



# Predict
y_pred_test = randomforest.predict(X_test)
y_pred_train = randomforest.predict(X_train)

# Evaluate
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)

print("Test MSE:", mse)
print("Test R²:", r2)
print("Train R²:", train_r2)

print(y_pred_test[:10])  # first 10 predictions

print(len(randomforest.estimators_))  # should equal n_estimators

