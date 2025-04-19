import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from murphet.churn_model import fit_churn_model

# Load the bike sharing dataset
df = pd.read_csv('/home/halsted/Documents/python/murphet/data/day_bikes.csv')

# Convert date column to datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# Calculate the ratio of casual to total bikes
df['casual_ratio'] = df['casual'] / df['cnt']

# Ensure ratio is between 0 and 1
assert df['casual_ratio'].between(0, 1).all()

# Create Prophet-compatible dataframe
prophet_df = df[['dteday', 'casual_ratio']].rename(columns={'dteday': 'ds', 'casual_ratio': 'y'})

# Split data for training and testing (80% train, 20% test)
train_size = int(len(df) * 0.8)
train_df = prophet_df[:train_size]
test_df = prophet_df[train_size:]

# Train Prophet model
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05
)
prophet_model.add_regressor('holiday')
prophet_model.add_regressor('temp')
prophet_model.add_regressor('windspeed')
prophet_model.fit(train_df.merge(df[['dteday', 'holiday', 'temp', 'windspeed']],
                                  left_on='ds', right_on='dteday').drop('dteday', axis=1))

# Create future dataframe for Prophet prediction
future_df = prophet_model.make_future_dataframe(periods=len(test_df))
future_df = future_df.merge(df[['dteday', 'holiday', 'temp', 'windspeed']],
                           left_on='ds', right_on='dteday', how='left').drop('dteday', axis=1)
prophet_forecast = prophet_model.predict(future_df)

# Prepare data for Murphet
t_train = np.arange(len(train_df))
y_train = train_df['y'].values
t_test = np.arange(len(train_df), len(df))
t_all = np.arange(len(df))

# Train Murphet model
murphet_model = fit_churn_model(
    t=t_train,
    y=y_train,
    num_harmonics=2,
    period=365.25,  # Annual seasonality
    n_changepoints=5,
    delta_scale=0.05,
    chains=1,
    iter=300,
    warmup=10
)

# Make predictions
murphet_predictions = murphet_model.predict(t_all)
prophet_train_preds = prophet_forecast['yhat'][:train_size].values
prophet_test_preds = prophet_forecast['yhat'][train_size:].values

# Calculate metrics on test set
murphet_rmse = np.sqrt(mean_squared_error(test_df['y'].values, murphet_predictions[train_size:]))
prophet_rmse = np.sqrt(mean_squared_error(test_df['y'].values, prophet_test_preds))
murphet_mae = mean_absolute_error(test_df['y'].values, murphet_predictions[train_size:])
prophet_mae = mean_absolute_error(test_df['y'].values, prophet_test_preds)

# Create visualization
plt.figure(figsize=(14, 10))
plt.style.use('ggplot')

# Plot actual data
plt.subplot(2, 1, 1)
plt.plot(df['dteday'], df['casual_ratio'], 'o-', color='black', alpha=0.6, markersize=3, label='Actual')
plt.plot(df['dteday'][:train_size], murphet_predictions[:train_size], '-', color='blue', label='Murphet (Train)')
plt.plot(df['dteday'][train_size:], murphet_predictions[train_size:], '--', color='blue',
         label=f'Murphet (Test, RMSE={murphet_rmse:.4f})')
plt.axvline(df['dteday'][train_size-1], color='gray', linestyle=':', lw=2)
plt.legend()
plt.title('Casual to Total Bike Ratio: Murphet Model', fontsize=14)
plt.ylabel('Casual/Total Ratio')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Prophet plot
plt.subplot(2, 1, 2)
plt.plot(df['dteday'], df['casual_ratio'], 'o-', color='black', alpha=0.6, markersize=3, label='Actual')
plt.plot(df['dteday'][:train_size], prophet_train_preds, '-', color='orange', label='Prophet (Train)')
plt.plot(df['dteday'][train_size:], prophet_test_preds, '--', color='orange',
         label=f'Prophet (Test, RMSE={prophet_rmse:.4f})')
plt.fill_between(
    df['dteday'][train_size:],
    prophet_forecast['yhat_lower'][train_size:],
    prophet_forecast['yhat_upper'][train_size:],
    color='orange', alpha=0.2, label='Prophet 80% CI'
)
plt.axvline(df['dteday'][train_size-1], color='gray', linestyle=':', lw=2)
plt.legend()
plt.title('Casual to Total Bike Ratio: Prophet Model', fontsize=14)
plt.ylabel('Casual/Total Ratio')
plt.xlabel('Date')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('bike_sharing_comparison.png', dpi=300)
plt.show()

# Print metrics
print("\n===== MODEL PERFORMANCE METRICS =====")
metrics_df = pd.DataFrame({
    "Model": ["Prophet", "Murphet"],
    "RMSE": [prophet_rmse, murphet_rmse],
    "MAE": [prophet_mae, murphet_mae]
})
print(metrics_df.set_index("Model").round(4))

# Additional seasonal/temporal analysis
plt.figure(figsize=(16, 10))

# By day of week
plt.subplot(2, 2, 1)
df['weekday'] = df['dteday'].dt.dayofweek
weekday_ratio = df.groupby('weekday')['casual_ratio'].mean()
plt.bar(weekday_ratio.index, weekday_ratio.values)
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.title('Average Casual Ratio by Day of Week')
plt.ylabel('Casual/Total Ratio')

# By month
plt.subplot(2, 2, 2)
df['month'] = df['dteday'].dt.month
month_ratio = df.groupby('month')['casual_ratio'].mean()
plt.bar(month_ratio.index, month_ratio.values)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title('Average Casual Ratio by Month')
plt.ylabel('Casual/Total Ratio')

# By temperature
plt.subplot(2, 2, 3)
plt.scatter(df['temp'], df['casual_ratio'], alpha=0.5)
plt.title('Casual Ratio vs Temperature')
plt.xlabel('Normalized Temperature')
plt.ylabel('Casual/Total Ratio')

# By weather
plt.subplot(2, 2, 4)
weather_ratio = df.groupby('weathersit')['casual_ratio'].mean()
plt.bar(weather_ratio.index, weather_ratio.values)
plt.xticks(range(1, 5), ['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain/Snow'])
plt.title('Average Casual Ratio by Weather')
plt.ylabel('Casual/Total Ratio')

plt.tight_layout()
plt.savefig('bike_sharing_analysis.png', dpi=300)
plt.show()