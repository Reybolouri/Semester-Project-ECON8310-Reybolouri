import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1) Load staffing grid
df = pd.read_csv("ecec_staffing_grouped.csv", parse_dates=["time_block"])

# 2) Filter to active students and aggregate total staff required per block
df_active = df[df["Student Status"] == "Active"]
ts = df_active.groupby("time_block")["staff_required"].sum().sort_index()

# --- Goal A: "Typical Week" Forecast via Weekly Averaging ---

# Extract day-of-week, hour, and minute
ts_df = ts.to_frame("staff_required")
ts_df["dow"]    = ts_df.index.dayofweek
ts_df["hour"]   = ts_df.index.hour
ts_df["minute"] = ts_df.index.minute

# Compute mean staff_required for each weekly time-slot
typical = (
    ts_df
    .groupby(["dow", "hour", "minute"])["staff_required"]
    .mean()
    .reset_index()
)

# Build a Timestamp index for one canonical week starting Monday
week_slots = []
week_values = []
for _, row in typical.iterrows():
    slot = pd.Timestamp("2025-01-06") \
         + pd.Timedelta(days=row["dow"]) \
         + pd.Timedelta(hours=row["hour"]) \
         + pd.Timedelta(minutes=row["minute"])
    week_slots.append(slot)
    week_values.append(row["staff_required"])

typical_week = pd.Series(week_values, index=week_slots).sort_index()

# Save typical-week profile
typical_week.to_csv("typical_week_staffing.csv", header=["staff_required"])

# --- Goal B: "Next Week" Forecast via SARIMAX ---

# SARIMAX with weekly seasonality: period = 7 days * 48 half-hour blocks = 336
model = SARIMAX(
    ts,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 336),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fit = model.fit(disp=False)

# Forecast the next 7 days at half-hour resolution (336 steps)
forecast = fit.get_forecast(steps=7 * 48)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Save next-week forecast
forecast_mean.to_csv("next_week_staffing_forecast.csv", header=["forecast"])
forecast_ci.to_csv("next_week_staffing_forecast_ci.csv")

print("✅ Saved 'typical_week_staffing.csv', 'next_week_staffing_forecast.csv', and CI file.")
