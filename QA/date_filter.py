import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as plt 
import plotly.graph_objects as go

def date_filter(
    df,
    date_col="datetime", #CLF format assumed
    air_col="t3", #CLF format assumed
    soil_col="t1", #CLF format assumed
    drop_threshold=-5,
    consecutive_nights=3,
    day_in_streak=2,
    buffer_days=7
):
    """
    Find cutoff date based on:
    1. Air temperature streak of nightly drops.
    2. (Optional) Soil temperature check vs baseline median if soil_col exists.

    Parameters
    ----------
    df : DataFrame
        Input dataframe with datetime and temperatures.
    date_col : str
        Column name for datetime.
    air_col : str
        Column name for air temperature.
    soil_col : str
        Column name for soil temperature (if exists, soil check will be applied).
    drop_threshold : float
        Required drop in air temp from max to min (negative, e.g. -5).
    consecutive_nights : int
        Number of consecutive days that will be chcked required.
    day_in_streak : int
        Which day of the streak to use as cutoff (1=first, 2=second, ...).
    buffer_days : int
        Gap in days between baseline and cutoff - this is required to avoid catching transitional data in the check.
    """
    # enforce datetime (CLF superfluous)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    # resample to daily and air temp stats
    daily_air = df[air_col].resample("1D").agg(["min", "max"])
    daily_air["drop"] = daily_air["min"] - daily_air["max"] #min max difference
    daily_air["condition"] = daily_air["drop"] <= drop_threshold #check if drop exceeds threshold
    daily_air["streak"] = daily_air["condition"].rolling(consecutive_nights).sum() #rolling window
    # streak finder
    streak_end = daily_air.index[daily_air["streak"] == consecutive_nights].min() #set streak end
    if pd.isna(streak_end):
        print("No cutoff found (no streak).")
        return df.reset_index(), None # fails if theres too little "office data"
    # define streak window and cutoff day candidate
    streak_start = streak_end - pd.Timedelta(days=consecutive_nights - 1)
    cutoff_day = streak_start + pd.Timedelta(days=day_in_streak - 1)
    # soil check if soil data is available
    if soil_col in df.columns and df[soil_col].notna().any():
        daily_soil = df[soil_col].resample("1D").median() #resample soil temp to median daily
        baseline_end = cutoff_day - pd.Timedelta(days=buffer_days) #define baseline
        baseline_series = daily_soil.loc[:baseline_end]
        if baseline_series.empty:
            print("Not enough data before cutoff for baseline.")
            return df.reset_index(), None
        #calculate median within baseline and soil streak
        baseline = baseline_series.median()
        soil_streak_median = daily_soil.loc[streak_start:streak_end].median()
        # if soil streak median is lower, pass check otherwise fail
        if soil_streak_median < baseline:
            print(f"Cutoff date confirmed: {cutoff_day.date()} "
                  f"(soil streak median {soil_streak_median:.2f} < baseline {baseline:.2f})")
            return df.loc[cutoff_day:].reset_index(), cutoff_day
        else:
            print(f"Soil check failed: streak median {soil_streak_median:.2f} "
                  f"not < baseline {baseline:.2f})")
            return df.reset_index(), None
    # fallback to air only if no data in soil column
    print(f"Cutoff date confirmed (air only): {cutoff_day.date()}")
    return df.loc[cutoff_day:].reset_index(), cutoff_day

df = pd.read_csv(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\clf_conversion\output\CLF_94200119_2024_11_04_TMS.csv")  # or however you load it
df
df_cleaned, cutoff_day = date_filter(
    df, date_col="datetime", air_col="t3", soil_col="t1",
    drop_threshold=-5, consecutive_nights=3, day_in_streak=2
)

#plt.figure(figsize=(12,5))
#plt.plot(df["datetime"], df["t3"], color="gray", alpha=0.5, label="Original")
#if cutoff_day:
#    plt.plot(df_cleaned["datetime"], df_cleaned["t3"], color="green", label="Filtered")
#    plt.axvline(cutoff_day, color="red", linestyle="--", label="Cutoff Day")
#plt.legend()
#plt.show()


fig = go.Figure()

df_cleaned
# Original (gray)
fig.add_trace(go.Scatter(x=df['datetime'], y=df['t3'],
                         mode='lines', name='Original',
                         line=dict(color='gray'), opacity=0.5))
# Filtered (green) + cutoff (red dashed)
fig.add_trace(go.Scatter(x=df_cleaned['datetime'], y=df_cleaned['t3'],
                             mode='lines', name='Filtered',
                             line=dict(color='green')))
fig.add_shape(type="line", x0=cutoff_day, x1=cutoff_day,
                  y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color="red", dash="dash"))   
fig.update_layout(title=f"test",
                  xaxis_title='Date',
                  yaxis_title='Temperature (°C)',
                  template='plotly_dark')

fig.write_html("plot.html")
import os
print(os.getcwd())

