import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from zipfile import ZipFile
import os

# Unzip and load dataset
zip_path = "/mnt/data/stock_details_5_years.csv.zip"
extract_path = "/mnt/data/unzipped_stock_data"
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

file_name = os.listdir(extract_path)[0]
df = pd.read_csv(os.path.join(extract_path, file_name))

# Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
aapl_df = df[df['Company'] == 'AAPL'].copy()
aapl_df.sort_values('Date', inplace=True)

# Financial Indicators
aapl_df['SMA_20'] = aapl_df['Close'].rolling(window=20).mean()
aapl_df['EMA_20'] = aapl_df['Close'].ewm(span=20, adjust=False).mean()

delta = aapl_df['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
aapl_df['RSI_14'] = 100 - (100 / (1 + rs))

aapl_df['BB_Middle'] = aapl_df['Close'].rolling(window=20).mean()
aapl_df['BB_Std'] = aapl_df['Close'].rolling(window=20).std()
aapl_df['BB_Upper'] = aapl_df['BB_Middle'] + 2 * aapl_df['BB_Std']
aapl_df['BB_Lower'] = aapl_df['BB_Middle'] - 2 * aapl_df['BB_Std']

# Drop NaN for modeling
aapl_clean = aapl_df.dropna().copy()

# Anomaly Detection with Isolation Forest
iso = IsolationForest(contamination=0.01, random_state=42)
aapl_clean['anomaly'] = iso.fit_predict(aapl_clean[['Close', 'SMA_20', 'EMA_20', 'RSI_14']])
aapl_clean['is_anomaly'] = aapl_clean['anomaly'] == -1

# Forecasting with Prophet
prophet_df = aapl_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).dropna()
model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Merge forecast with actual data
forecast_result = pd.merge(prophet_df, forecast[['ds', 'yhat']], on='ds', how='left')
forecast_result['error'] = np.abs(forecast_result['y'] - forecast_result['yhat'])

# Visualization
plt.figure(figsize=(14, 6))
plt.plot(aapl_clean['Date'], aapl_clean['Close'], label='Close Price')
plt.scatter(aapl_clean[aapl_clean['is_anomaly']]['Date'],
            aapl_clean[aapl_clean['is_anomaly']]['Close'],
            color='red', label='Anomaly', s=30)
plt.title('AAPL Stock Price with Detected Anomalies')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: View forecast
model.plot(forecast)
plt.title('AAPL Stock Price Forecast with Prophet')
plt.tight_layout()
plt.show()
