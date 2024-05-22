import yfinance as yf  # Replace with actual library if using a different API
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def get_historical_data(ticker, period="1y"):
  # Download historical stock data using the financial data API
  data = yf.download(ticker, period=period)
  return data

def prepare_data(data):
  # Select relevant features (e.g., closing price) and scale the data
  closing_prices = data["Close"]
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(closing_prices.values.reshape(-1, 1))
  
  # Split data into training and testing sets
  training_size = int(len(scaled_data) * 0.8)
  training_data, testing_data = scaled_data[0:training_size], scaled_data[training_size:]
  
  return training_data, testing_data, scaler

def train_model(training_data):
  # Separate features (X) and target values (y)
  X_train, y_train = training_data[:, :-1], training_data[:, -1]
  
  # Create and train a machine learning model (e.g., Linear Regression)
  model = LinearRegression()
  model.fit(X_train, y_train)
  return model

def predict_price(model, testing_data, scaler, future_day=1):
  # Get the closing price for the previous day
  previous_day_price = scaler.inverse_transform([[testing_data[-1]]])[0][0]
  
  # Predict the closing price for the future day using the model
  predicted_price = model.predict([[previous_day_price]])[0][0]
  
  # Unscale the predicted price
  unscaled_price = scaler.inverse_transform([[predicted_price]])[0][0]
  return unscaled_price

def main():
  # Get user input for stock ticker symbol
  ticker = input("Enter stock ticker symbol: ")
  
  # (Optional) Get user input for time frame
  period = input("Enter time frame for historical data (optional, default: 1y): ") or "1y"
  
  # Download historical data
  data = get_historical_data(ticker, period)
  
  # Prepare data for training
  training_data, testing_data, scaler = prepare_data(data)
  
  # Train the model
  model = train_model(training_data)
  
  # Predict closing price
  predicted_price = predict_price(model, testing_data, scaler)
  print(f"Predicted closing price for tomorrow (or {period} after the last data point): {predicted_price:.2f}")

if __name__ == "__main__":
  main()
