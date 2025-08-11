# bike_rental_prediction.py

# 1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load Dataset
df = pd.read_csv("day.csv")  # use path to your dataset
print("Dataset Shape:", df.shape)
print(df.head())

# 3. Feature Selection
# Drop unnecessary columns
df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

# Target variable
target = 'cnt'

# 4. Train-Test Split
X = df.drop([target], axis=1)
y = df[target]

# Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Training

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# --- Random Forest Regressor ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# 6. Evaluation Function
def evaluate_model(true, predicted, name):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2 = r2_score(true, predicted)
    print(f"\n{name} Model Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return mae, rmse, r2

# 7. Evaluate Both Models
evaluate_model(y_test, lr_preds, "Linear Regression")
evaluate_model(y_test, rf_preds, "Random Forest")

# 8. Visualization - Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', linewidth=2)
plt.plot(rf_preds, label='Random Forest Prediction', alpha=0.7)
plt.title('Actual vs Predicted Rentals')
plt.xlabel('Test Sample Index')
plt.ylabel('Rental Count')
plt.legend()
plt.tight_layout()
plt.savefig('visuals/actual_vs_predicted.png')
plt.show()

# 9. Feature Importance (Random Forest)
importances = rf.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Bar Plot
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('visuals/feature_importance.png')
plt.show()
