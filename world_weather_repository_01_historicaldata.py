import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("/content/GlobalWeatherRepository_kaggle.csv")

data.size

numeric_columns_count = data.select_dtypes(include='number').shape[1]
categorical_columns_count = data.select_dtypes(include='object').shape[1]

print(f'Numeric columns: {numeric_columns_count}')
print(f'Categorical columns: {categorical_columns_count}')
print(f'Shape of the Dataset: {data.shape}')

"""# --- Exploratory Data Analysis (EDA) ---"""

data.info()  # Check for null values and data type

data.describe().T # Statistical summary

missing_values=data.isnull().sum()
print("Missing values:\n")
print(missing_values)
missing_values = data.isnull().sum().sum()
# Check if there are any missing values and print the result using an f-string
if missing_values > 0:
    print(f"Total missing values: {missing_values}")
    data = data.dropna()
    print("Missing values is deleted")
else:
    print(f"No missing values in the Dataset.")

duplicates_count = data.duplicated().sum()
# Check if there are any duplicate rows and print the result using f-strings
if data.duplicated().any():
    print(f"Duplicates are present. Total duplicate rows: {duplicates_count}")
    data = data.drop_duplicates()
    print("Duplicates values is deleted")
else:
    print(f"No duplicates are present in the Dataset.")

data.columns

data['country'].nunique()

data['country'].unique()

"""
# Visualize distributions of key features"""

sns.histplot(data["temperature_celsius"], bins=30, kde=True)
plt.title("Temperature Distribution")
plt.show()

"""# Correlation heatmap"""

# Correlation heatmap for numeric features
plt.figure(figsize=(20, 25))
sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations (Numeric Columns Only)")
plt.show()

"""# --- Data Preprocessing ---"""

# Handle missing values
# data.fillna(data.mean(), inplace=True)

drop_columns = [
    "country", "location_name", "timezone", "last_updated_epoch", "last_updated",
    "temperature_fahrenheit", "condition_text", "pressure_in", "precip_in",
    "feels_like_celsius", "feels_like_fahrenheit", "visibility_miles", "gust_kph",
    "sunrise", "sunset", "moonrise", "moonset", "moon_phase", "moon_illumination"
]

data.drop(columns=[col for col in drop_columns if col in data.columns], inplace=True)

data.shape

data.columns

data.info()

data.drop(columns=["wind_direction",  "wind_kph"], inplace=True)

data.shape

selected_features = [
    "latitude", "longitude", "humidity", "cloud", "pressure_mb", "precip_mm",
    "wind_mph", "wind_degree", "visibility_km", "uv_index", "gust_mph",
    "air_quality_Carbon_Monoxide", "air_quality_Ozone", "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide", "air_quality_PM2.5", "air_quality_PM10",
    "air_quality_us-epa-index", "air_quality_gb-defra-index"
]
len(selected_features)

# Feature selection
X = data.drop(columns=["temperature_celsius"])  # Features
y = data["temperature_celsius"]  # Target

X.shape

y.shape

"""# Standardize numerical features"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""# --- Splitting Data ---"""

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

"""# --- Model Training ---

# RandomForestRegressor
"""

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

"""# --- Model Evaluation ---"""

y_pred = rf_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

"""# XGBoost Regressor"""

from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("MSE:", mean_squared_error(y_test, y_pred_xgb))
print("R2 Score:", r2_score(y_test, y_pred_xgb))

import matplotlib.pyplot as plt
import seaborn as sns

# Metrics for both models
metrics = ["MAE", "MSE", "R2 Score"]
rf_values = [1.4419, 4.5440, 0.9507]  # Random Forest values
xgb_values = [1.6993, 5.7347, 0.9378]  # XGBoost values

# Create DataFrame
import pandas as pd
df = pd.DataFrame({"Metric": metrics, "RandomForest": rf_values, "XGBoost": xgb_values})

# Melt DataFrame for Seaborn
df_melted = df.melt(id_vars="Metric", var_name="Model", value_name="Value")

df.head()

df_melted.head()

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Metric", y="Value", hue="Model", data=df_melted, palette="coolwarm")
plt.title("Comparison of RandomForest and XGBoost Performance")
plt.show()

"""# --- Anomaly Detection ---

Isolation Forest is an **ensemble learning method** based on **decision trees**. It isolates anomalies by recursively splitting data points. Anomalies get isolated faster (fewer splits) than normal points, making them easier to detect. It is a type of **unsupervised learning** algorithm.
"""

anomaly_detector = IsolationForest(contamination=0.02, random_state=42)
anomaly_detector.fit(X_train)
data["anomaly"] = anomaly_detector.predict(X_scaled)
data["anomaly"] = data["anomaly"].apply(lambda x: 1 if x == -1 else 0)

data.shape

data.value_counts("anomaly")

data.head()



"""
# --- Hyperparameter Tuning ---"""

# param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)   # K-fold
# grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=kfold, scoring="r2")
# grid_search.fit(X_train, y_train)
# print("Best Parameters:", grid_search.best_params_)

"""# --- Save Model ---"""

joblib.dump(rf_model, "weather_prediction_model.pkl")
joblib.dump(anomaly_detector, "anomaly_detection_model.pkl")
print("Models saved successfully.")