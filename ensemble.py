
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# If running in Colab, upload your cleaned CSV:
# from google.colab import files
# uploaded = files.upload()  # then select ecec_staffing_grouped.csv
# df = pd.read_csv("ecec_staffing_grouped.csv", parse_dates=["time_block"])

# Or, if CSV is already in the working directory:
df = pd.read_csv("ecec_staffing_grouped.csv", parse_dates=["time_block"])

# Aggregate to total staff required per half-hour block
ts = (
    df[df["Student Status"]=="Active"]
      .groupby("time_block")["staff_required"]
      .sum()
      .sort_index()
)

# Resample to perfect 30-minute grid and fill gaps
ts = ts.resample("30min").sum().interpolate()

# Train/Test split: last calendar week as hold-out
H = 7 * 48
train, test = ts.iloc[:-H], ts.iloc[-H:]

#  Feature Engineering for RF & XGB ===
def make_features(index):
    df_feat = pd.DataFrame({"ds": index})
    df_feat["dow"]    = df_feat["ds"].dt.dayofweek
    df_feat["hour"]   = df_feat["ds"].dt.hour
    df_feat["minute"] = df_feat["ds"].dt.minute
    # Cyclical encoding
    df_feat["dow_sin"]  = np.sin(2*np.pi*df_feat["dow"]/7)
    df_feat["dow_cos"]  = np.cos(2*np.pi*df_feat["dow"]/7)
    df_feat["hour_sin"] = np.sin(2*np.pi*df_feat["hour"]/24)
    df_feat["hour_cos"] = np.cos(2*np.pi*df_feat["hour"]/24)
    df_feat["min_sin"]  = np.sin(2*np.pi*df_feat["minute"]/60)
    df_feat["min_cos"]  = np.cos(2*np.pi*df_feat["minute"]/60)
    return df_feat[["dow_sin","dow_cos","hour_sin","hour_cos","min_sin","min_cos"]]

X_train = make_features(train.index)
y_train = train.values
X_test  = make_features(test.index)

#  Train Random Forest & XGBoost
# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost
xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, objective="reg:squarederror")
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

#Train Prophet 
# Prepare dataframe for Prophet
prophet_df = train.reset_index().rename(columns={"time_block":"ds","staff_required":"y"})
m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
m.fit(prophet_df)

# Build future dataframe
future = m.make_future_dataframe(periods=H, freq="30min")
forecast = m.predict(future)
prophet_pred = forecast.set_index("ds")["yhat"].loc[test.index]

#  Accuracy Metrics & Plot
models = {
    "RandomForest": rf_pred,
    "XGBoost":      xgb_pred,
    "Prophet":      prophet_pred.values
}

# Compute and print accuracy
for name, pred in models.items():
    mae  = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    mape = np.mean(np.abs((pred - test) / test)) * 100
    print(f"{name:12s} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")

# Plot actual vs predictions
plt.figure(figsize=(12,4))
plt.plot(test.index, test, label="Actual", color="black")
plt.plot(test.index, rf_pred,  label="RandomForest")
plt.plot(test.index, xgb_pred, label="XGBoost")
plt.plot(test.index, prophet_pred, label="Prophet")
plt.legend()
plt.title("Next-Week Forecast: Actual vs Models")
plt.xlabel("Time")
plt.ylabel("Staff Required")
plt.show()
