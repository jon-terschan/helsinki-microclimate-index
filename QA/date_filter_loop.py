import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as plt 
import plotly.graph_objects as go

#####################
# SOURCE & SETTINGS #
#####################
source_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\2_dategate\source" # input path fll with .csv
output_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\2_dategate\output" # output path for CLF.csvs
figure_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\2_dategate\figures" # output path for figures

#####################
# FUNCTION DEF #
#####################
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
    return df.loc[df.index > cutoff_day].reset_index(), cutoff_day

#####################
# LOOP FUNCTION #
#####################
for filename in os.listdir(source_folder):
      if filename.endswith(".csv") and ("TMS" in filename.upper() or "TL" in filename.upper()):
          file_path = os.path.join(source_folder, filename)
          df = pd.read_csv(file_path, sep = ",", dayfirst = True)
          df_filtered, cutoff_day = date_filter(
            df, date_col="datetime",
            air_col="t3",
            soil_col="t1",
            drop_threshold=-5,
            consecutive_nights=3,
            day_in_streak=3
            )
          new_filename = f"{filename[:-4]}_filtered.csv"
          new_file_path = os.path.join(output_folder, new_filename)
          print(df_filtered.head())
          print(df_filtered.dtypes)
          df_filtered.to_csv(new_file_path, index=False)
          print(f"{filename}: date filtered exported, proceed to Figure!")
          # FIGURES
          fig = go.Figure()
          fig.add_trace(go.Scatter(x=df['datetime'], y=df['t3'], #original air data
                         mode='lines', name='Original T air',
                         line=dict(color='gray'), opacity=0.5))
          fig.add_trace(go.Scatter(x=df_filtered['datetime'], y=df_filtered['t3'], #filtered air data
                             mode='lines', name='Filtered T air',
                             line=dict(color='green')))
          if 'TMS' in filename.upper(): # add other data if TMS
            fig.add_trace(go.Scatter(x=df_filtered['datetime'], y=df_filtered['t2'], mode='lines', name='T surface'))
            fig.add_trace(go.Scatter(x=df_filtered['datetime'], y=df_filtered['t1'], mode='lines', name='T soil'))
          fig.add_shape(type="line", x0=cutoff_day, x1=cutoff_day, #cutoff
                  y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color="red", dash="dash"))    
          fig.update_layout(
              
              
              title=f"{'Air temperature' if 'TL' in filename.upper() else 'Soil, surface, air temperature'}, {filename}",
              xaxis_title='Date',
              yaxis_title='Temperature (°C)',
              template='plotly_dark')
          figure_name = f"{filename[:-4]}_filtered_fig.html"
          figure_path = os.path.join(figure_folder, figure_name)
          fig.write_html(figure_path)
          print(f"{filename}: Figure exported, proceed to next!")