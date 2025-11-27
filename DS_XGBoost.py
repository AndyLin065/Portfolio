import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

# Dataset: House Prices - Advanced Regression
# Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# Alternative: California Housing from sklearn

from sklearn.datasets import fetch_california_housing

# Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseValue')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with important hyperparameters
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1,   # L2 regularization
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"R² Score: {model.score(X_test, y_test):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:\n", feature_importance)

# Plot feature importance
xgb.plot_importance(model, max_num_features=10)
plt.show()

# Plot training progress
results = model.evals_result()
plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['rmse'], label='Train')
plt.plot(results['validation_1']['rmse'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('RMSE')
plt.legend()
plt.title('Training Progress')
plt.show()
