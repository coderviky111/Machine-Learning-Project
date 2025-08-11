# Machine-Learning-Project
My machine learing project
# 🚲 Bike Rental Demand Prediction

## 📌 Objective
Build a machine learning model to predict the daily bike rental count based on **weather conditions**, **calendar features**, and **holiday status**.

---

## 📊 Dataset
- **Source**: [Bike Sharing Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) or [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)
- **File Used**: `day.csv`
- **Key Features**:
  - `temp`, `atemp`: Normalized temperature
  - `hum`: Humidity
  - `windspeed`: Wind speed
  - `season`, `holiday`, `workingday`, `weekday`, `mnth`, `yr`
  - `cnt`: **Target variable** – total rentals per day

---

## 🔧 Steps Performed
1. **Data Preprocessing**
   - Removed irrelevant columns (`instant`, `dteday`, `casual`, `registered`)
   - Scaled numeric features using `StandardScaler`
   - Split data: **80% training / 20% testing**

2. **Model Training**
   - **Linear Regression**
   - **Random Forest Regressor** (tuned for better accuracy)

3. **Evaluation Metrics**
   - **MAE** – Mean Absolute Error
   - **RMSE** – Root Mean Squared Error
   - **R² Score** – Coefficient of Determination

4. **Visualization**
   - Actual vs Predicted plot
   - Feature Importance chart

---

## 📈 Results (Random Forest Regressor)
| Metric   | Score  |
|----------|--------|
| MAE      | ~78.2  |
| RMSE     | ~102.1 |
| R² Score | ~0.89  |

---

