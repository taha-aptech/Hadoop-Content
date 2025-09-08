import streamlit as st
import bcrypt
import jwt
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Secret key for JWT encoding and decoding (should be kept safe in a real-world app)
SECRET_KEY = "your_secret_key_here"

# Hardcoded users (In real applications, store these securely, e.g., in a database)
users = {
    "admin": {
        "password": bcrypt.hashpw("adminpassword".encode(), bcrypt.gensalt()).decode(),
        "role": "admin"
    },
    "analyst": {
        "password": bcrypt.hashpw("analystpassword".encode(), bcrypt.gensalt()).decode(),
        "role": "analyst"
    }
}

# Function to create JWT token
def create_jwt_token(username, role):
    payload = {
        "username": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # 1 hour expiration
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

# Function to validate JWT token
def validate_jwt_token(token):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Function to authenticate user and generate token
def authenticate_user(username, password):
    if username in users and bcrypt.checkpw(password.encode(), users[username]["password"].encode()):
        token = create_jwt_token(username, users[username]["role"])
        return token
    return None

# Function to get user role from JWT token
def get_role_from_token(token):
    decoded = validate_jwt_token(token)
    if decoded:
        return decoded["role"]
    return None

# Streamlit app interface
def main():
    st.title("Global Temperature Prediction App")

    # Check if a token is stored in the session
    token = st.session_state.get("token", None)

    # If no token is found, show login form
    if not token:
        login_form()

    else:
        # If user is logged in, show content based on their role
        role = st.session_state.role
        username = st.session_state.username
        st.sidebar.write(f"Welcome, {username} ({role.capitalize()})")

        # Role-based access
        if role == "admin":
            show_full_data_analysis()
        elif role == "analyst":
            show_predictions()

        if st.button("Logout"):
            del st.session_state.token
            del st.session_state.username
            del st.session_state.role
            st.success("Logged out successfully!")
            st.session_state.clear()  # Use this to clear session state and refresh

def login_form():
    """Render login form"""
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Authenticate the user
        token = authenticate_user(username, password)
        if token:
            st.session_state.token = token
            st.session_state.username = username
            st.session_state.role = get_role_from_token(token)
            st.success("Logged in successfully!")
            st.session_state["login"] = True  # You can trigger rerun implicitly by updating session_state
        else:
            st.error("Invalid credentials. Please try again.")

def show_full_data_analysis():
    """Render data analysis and visualizations for admin"""
    st.write("### Global Average Temperature Over Time")
    # Load and preprocess the dataset
    data = pd.read_csv('GlobalTemperatures.csv')
    data.dropna(subset=['LandAverageTemperature'], inplace=True)
    data['dt'] = pd.to_datetime(data['dt'])
    data.set_index('dt', inplace=True)
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    st.line_chart(data['LandAverageTemperature'])

    # Train-test split
    train_size = int(0.8 * len(data))
    train, test = data[:train_size], data[train_size:]
    X_train = train[['Year', 'Month']]
    y_train = train['LandAverageTemperature']
    X_test = test[['Year', 'Month']]
    y_test = test['LandAverageTemperature']

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Exponential Smoothing Model
    exp_model = ExponentialSmoothing(train['LandAverageTemperature'], seasonal='add', seasonal_periods=12)
    exp_model_fit = exp_model.fit()
    y_pred_exp = exp_model_fit.forecast(len(test))
    mse_exp = mean_squared_error(test['LandAverageTemperature'], y_pred_exp)
    r2_exp = r2_score(test['LandAverageTemperature'], y_pred_exp)

    # Display results
    st.write(f"Linear Regression MSE: {mse_lr:.4f}")
    st.write(f"Linear Regression R²: {r2_lr:.4f}")
    st.write(f"Exponential Smoothing MSE: {mse_exp:.4f}")
    st.write(f"Exponential Smoothing R²: {r2_exp:.4f}")

    # Plot actual vs predicted temperatures
    st.write("### Actual vs Predicted Temperatures")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(test.index, test['LandAverageTemperature'], label='Actual Temperature', color='orange')
    ax.plot(test.index, y_pred_lr, label='Linear Regression Prediction', color='blue')
    ax.plot(test.index, y_pred_exp, label='Exponential Smoothing Prediction', color='green')
    ax.set_xlabel('Year')
    ax.set_ylabel('Land Average Temperature (°C)')
    ax.set_title('Temperature Prediction: Actual vs Predicted')
    ax.legend()
    st.pyplot(fig)

def show_predictions():
    """Render data and predictions for analysts"""
    st.write("### Predictions for Analysts")
    # Load the dataset for predictions only
    data = pd.read_csv('GlobalTemperatures.csv')
    data.dropna(subset=['LandAverageTemperature'], inplace=True)
    data['dt'] = pd.to_datetime(data['dt'])
    data.set_index('dt', inplace=True)
    data['Year'] = data.index.year
    data['Month'] = data.index.month

    # Train-test split
    train_size = int(0.8 * len(data))
    train, test = data[:train_size], data[train_size:]
    X_train = train[['Year', 'Month']]
    y_train = train['LandAverageTemperature']
    X_test = test[['Year', 'Month']]
    y_test = test['LandAverageTemperature']

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Exponential Smoothing Model
    exp_model = ExponentialSmoothing(train['LandAverageTemperature'], seasonal='add', seasonal_periods=12)
    exp_model_fit = exp_model.fit()
    y_pred_exp = exp_model_fit.forecast(len(test))

    # Plot actual vs predicted temperatures
    st.write("### Actual vs Predicted Temperatures")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(test.index, test['LandAverageTemperature'], label='Actual Temperature', color='orange')
    ax.plot(test.index, y_pred_lr, label='Linear Regression Prediction', color='blue')
    ax.plot(test.index, y_pred_exp, label='Exponential Smoothing Prediction', color='green')
    ax.set_xlabel('Year')
    ax.set_ylabel('Land Average Temperature (°C)')
    ax.set_title('Temperature Prediction: Actual vs Predicted')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
