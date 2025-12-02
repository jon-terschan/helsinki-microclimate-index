import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# buncha inputs
SOURCE_FOLDER = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\4_finetuning\source"
OUT_FOLDER = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\4_finetuning\output"
FIG_FOLDER = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\4_finetuning\figures"
ipol_limit = 20 # how many entries can the gap be 

################################################################################################################
# DATA HANDLING FUNCTIONS
################################################################################################################
# find file path in source folder by serial number
def find_file_by_serial(source_folder, serial):
    pattern = re.compile(rf"CLF_{serial}_.+\.csv$", re.IGNORECASE)
    for file in os.listdir(source_folder):
        if pattern.match(file):
            return os.path.join(source_folder, file)
    raise FileNotFoundError(f"No {serial} in {source_folder}")

# loads .csv, checks integrity
def load_file(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values("datetime")
    print(f"Loaded {len(df)} rows from {path}.")
    return df

# export file
def export_cleaned(df, input_file_name, out_folder):
    cleaned_name = input_file_name.replace(".csv", "_edited.csv")
    out_path = os.path.join(out_folder, cleaned_name)
    df.to_csv(out_path, index=False)
    print(f"Exported cleaned file → {out_path}")
    return out_path

# detect logger system
def detect_logger_type(df):
    if df[['t1','t2','SMC']].isna().all().all():
        return "TL"
    return "TMS"

# create my beautiful figure mwah mwah
def create_figure(df, cleaned_name, figure_folder):
    logger_type = detect_logger_type(df)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['t3'], mode='lines', name='T air'
    ))

    if logger_type == "TMS":
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['t2'], mode='lines', name='T surface'))
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['t1'], mode='lines', name='T soil'))

    fig.update_layout(
        title=f"Edited: {cleaned_name}",
        template='plotly_dark',
        hovermode='x unified',
        xaxis_title='Date',
        yaxis_title='Temperature (°C)',
        height=700,
        width=1200
    )

    fig_path = os.path.join(
        figure_folder,
        cleaned_name.replace(".csv", "_FIG.html")
    )
    fig.write_html(fig_path)
    print(f"Figure saved → {fig_path}")

    return fig_path

def flag_out_of_soil_span(df, start, end):
    """
    Mark a datetime span as out-of-soil (OOS=1).
    Creates column 'OOS' if it does not exist.
    """
    start = pd.to_datetime(start, utc=True)
    end   = pd.to_datetime(end,   utc=True)

    # ensure the column exists
    if "OOS" not in df.columns:
        df["OOS"] = 0

    mask = (df['datetime'] >= start) & (df['datetime'] <= end)
    df.loc[mask, "OOS"] = 1

    print(f"Flagged {mask.sum()} rows as Out-of-Soil (OOS=1).")
    return df


####################################################
# DATA EDITING FUNCTIONS
####################################################
def remove_values(df, col, timestamps):
    ts = pd.to_datetime(timestamps, utc=True)
    mask = df['datetime'].isin(ts)
    df.loc[mask, col] = np.nan
    print(f"Removed {mask.sum()} values from {col}.")
    return df

def remove_values_span(df, col, start, end, flag_oos=False):
    start = pd.to_datetime(start, utc=True)
    end   = pd.to_datetime(end,   utc=True)
    mask  = (df['datetime'] >= start) & (df['datetime'] <= end)

    # set column values to NaN
    df.loc[mask, col] = np.nan
    print(f"Removed {mask.sum()} entries in {col} from {start} to {end}.")

    # optionally flag out-of-soil
    if flag_oos:
        if "OOS" not in df.columns:
            df["OOS"] = 0
        df.loc[mask, "OOS"] = 1
        print(f"Flagged {mask.sum()} rows as Out-of-Soil (OOS=1).")

    return df

def remove_rows(df, timestamps=None, start=None, end=None, by_date=False):
    # this is some vibecoding github copilot shit, i havent checked if it works yet
    """
    Remove rows from df in several flexible ways.
    - timestamps: scalar or list of strings/Timestamps
        * If any timestamp string contains only a date (YYYY-MM-DD) AND by_date=True,
          it will remove all rows on that calendar date.
        * If timestamp string includes time, it will match exact datetimes (tz-aware).
    - start, end: remove rows where start <= datetime <= end (both inclusive).
      Strings will be parsed as UTC timestamps (if date-only, they select midnight).
    - by_date: if True, treat date-only inputs as date matches (removes entire day).
    """
    df = df.copy()
    before = len(df)
    mask_remove = pd.Series(False, index=df.index)

    # 1) timestamps (scalar -> list)
    if timestamps is not None:
        if not isinstance(timestamps, (list, tuple, set, pd.Series)):
            timestamps = [timestamps]

        for t in timestamps:
            # parse
            ts = pd.to_datetime(t, utc=True)

            # If original input was date-only string (no time) AND by_date True,
            # remove all rows that have the same calendar date in UTC.
            if by_date and (isinstance(t, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}$", t)):
                date_only = ts.date()
                mask = df['datetime'].dt.date == date_only
                mask_remove |= mask
            else:
                # exact timestamp match (tz-aware)
                mask = df['datetime'] == ts
                mask_remove |= mask

    # 2) start/end range removal
    if start is not None or end is not None:
        if start is not None:
            start_ts = pd.to_datetime(start, utc=True)
        else:
            start_ts = df['datetime'].min()
        if end is not None:
            end_ts = pd.to_datetime(end, utc=True)
        else:
            end_ts = df['datetime'].max()

        mask_range = (df['datetime'] >= start_ts) & (df['datetime'] <= end_ts)
        mask_remove |= mask_range

    removed_count = mask_remove.sum()
    if removed_count > 0:
        df = df.loc[~mask_remove].copy()
    else:
        print("No rows matched the removal criteria.")

    print(f"Removed {removed_count} rows (before: {before}, after: {len(df)}).")
    return df

# interpolate missing data
def interpolate_missing(df, limit): 
    #second argument = how many entries large the gap can be 
    cols = ['t1','t2','t3','SMC']
    before = df[cols].isna().sum().sum()

    df[cols] = df[cols].interpolate(
        method='linear',
        limit=limit,
        limit_direction='both'
    )

    after = df[cols].isna().sum().sum()
    print(f"Filled {before - after} missing values via interpolation.")
    return df


##########################################################################################
# MASTER FUNCTION
##########################################################################################
def edit_file(serial, source_folder, output_folder, figure_folder, edits_fn):
    source_path = find_file_by_serial(source_folder, serial) # find file
    df = load_file(source_path) # load
    df = edits_fn(df) # apply edits
    cleaned_path = export_cleaned(df, os.path.basename(source_path), output_folder) # export
    create_figure(df, os.path.basename(cleaned_path), figure_folder) # create fig
    print("\nEDIT COMPLETE\n") # just FYI

# NESTED MASTER FUNCTION SETTINGS
def my_edits(df):
    #df = remove_values(df, "t1", ["2024-07-01 00:00"])
    df = remove_values_span(df, "t1", "2024-06-01", "2024-09-26", flag_oos=True)
    #df = remove_values_span(df, "t1", "2024-07-31", "2024-09-23", flag_oos=True)
    #df = remove_rows(df, start="2024-01-01", end="2024-04-26")
    #df = interpolate_missing(df, 20)
    return df

## DOCUMENTATION 
## remove all rows on April 25, 2024 (any time that day, UTC)
#df = remove_rows(df, timestamps="2024-04-25", by_date=True)
# remove a single exact timestamp
#df = remove_rows(df, timestamps="2024-04-25 00:15", by_date=False)
# remove several dates/timestamps
#df = remove_rows(df, timestamps=["2024-04-25","2024-05-01"], by_date=True)
# remove a range
#df = remove_rows(df, start="2024-04-24", end="2024-04-26")

### EXECUTE
edit_file(
    serial="94290008",
    source_folder=SOURCE_FOLDER,
    output_folder=OUT_FOLDER,
    figure_folder=FIG_FOLDER,
    edits_fn=my_edits
)


