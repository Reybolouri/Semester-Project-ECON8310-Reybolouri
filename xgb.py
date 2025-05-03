import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1) Load the staffing grid
df = pd.read_csv("ecec_staffing_grouped.csv", parse_dates=["time_block"])

# 2) Build the target series: total staff required per block
ts = (
    df[df["Student Status"]=="Active"]
      .groupby("time_block")["staff_required"]
      .sum()
      .sort_index()
)

# 3) Create a features DataFrame
data = ts.reset_index().rename(columns={"time_block":"ds","staff_required":"y"})
data["dow"]    = data["ds"].dt.dayofweek
data["hour"]   = data["ds"].dt.hour
data["minute"] = data["ds"].dt.minute
# Cyclical encoding for time features
data["dow_sin"]    = np.sin(2*np.pi*data["dow"]/7)
data["dow_cos"]    = np.cos(2*np.pi*data["dow"]/7)
data["hour_sin"]   = np.sin(2*np.pi*data["hour"]/24)
data["hour_cos"]   = np.cos(2*np.pi*data["hour"]/24)
data["min_sin"]    = np.sin(2*np.pi*data["minute"]/60)
data["min_cos"]    = np.cos(2*np.pi*data["minute"]/60)

# 4) Train/test split: last week as test
h = 7 * 48
train_df = data.iloc[:-h]
test_df  = data.iloc[-h:]

X_train = train_df[["dow_sin","dow_cos","hour_sin","hour_cos","min_sin","min_cos"]]
y_train = train_df["y"]
X_test  = test_df[ ["dow_sin","dow_cos","hour_sin","hour_cos","min_sin","min_cos"]]
y_test  = test_df["y"]

# 5) Fit XGBoost regressor
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)
model.fit(X_train, y_train)

# 6) Predict & compute accuracy
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

print("XGBoost hold-out accuracy:")
print(f"MAE   = {mae:.2f}")
print(f"RMSE  = {rmse:.2f}")
print(f"MAPE% = {mape:.1f}%")

# 7) Save the next-week predictions vs actual
pd.Series(y_pred, index=test_df["ds"]).to_csv("next_week_xgb_pred.csv", header=["y_pred"])
pd.Series(y_test.values, index=test_df["ds"]).to_csv("actual_week.csv",   header=["y_actual"])