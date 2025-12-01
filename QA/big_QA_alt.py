import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

###################################
# FILEPATHS
###################################
SOURCE_FOLDER = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\3_QA\source"
OUT_FOLDER = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\3_QA\output"
REP_FOLDER = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\3_QA\reports"
FIG_FOLDER = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\3_QA\figures" 

###################################
# DETECT LOGGER TYPE
###################################
# this is only useful for the report, and it only works if file are in CLF
def detect_logger_type(df):
    # TL = t1, t2, SMC all NaN
    if df[['t1','t2','SMC']].isna().all().all():
        return "TL"
    return "TMS"

###################################
# 0. DETECT & INSERT MISSING TIMESTAMPS
###################################
def add_missingentries(df, freq="15min", small_gap_limit=20):
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime')

    full_range = pd.date_range(
        start=df['datetime'].min(),
        end=df['datetime'].max(),
        freq=freq,
        tz="UTC"
    )

    # Reindex → inserts missing timestamps as rows with NaN values
    df_full = df.set_index('datetime').reindex(full_range)
    df_full.index.name = 'datetime'
    df_full = df_full.reset_index()

    # Count how many timestamps were inserted
    inserted = df_full.isna().all(axis=1).sum()

    # --- Detect large gaps (for reporting) ---
    gaps = []
    time_diffs = df['datetime'].diff()

    for i in range(1, len(time_diffs)):
        if time_diffs.iloc[i] > pd.Timedelta(freq):
            gap_minutes = time_diffs.iloc[i] / pd.Timedelta("1 minute")
            gap_size = int(gap_minutes // 15)

            if gap_size > small_gap_limit:
                gaps.append({
                    "start": df['datetime'].iloc[i-1],
                    "end": df['datetime'].iloc[i],
                    "missing_intervals": gap_size
                })

    return df_full, inserted, gaps

###################################
# 1. ENFORCE 15-MINUTE TIMESTAMPS
###################################
# delete non 15-mins timestamps because they will be faulty
def enforce_15min(df):
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    mask = df['datetime'].dt.minute % 15 == 0
    removed = len(df) - mask.sum()
    return df[mask].copy(), removed

###################################
# 2. REMOVE INSANE VALUES & JUMPS
###################################
def remove_insane_and_jumps(df):
    # sensible temp ranges for study area, here helsinki 
    ranges = {
        't1': (-20, 30),   # soil
        't2': (-35, 45),   # surface
        't3': (-35, 40),   # air
    }
    # max allowed temperature jump between entries
    # entries are 15 minutes, could probably be even more conservative
    jumps = {
        't1': 3,   # soil
        't2': 4,  # surface
        't3': 4,   # air
    }
    
    insane_count = 0
    jump_count = 0
    for col in ['t1', 't2', 't3']:

        valid_min, valid_max = ranges[col]
        max_jump = jumps[col]

        # Impossible absolute values
        insane_mask = (df[col] < valid_min) | (df[col] > valid_max)
        insane_count += insane_mask.sum()
        df.loc[insane_mask, col] = np.nan

        # Step jumps > allowed threshold
        diffs = df[col].diff().abs()
        jump_mask = diffs > max_jump
        jump_count += jump_mask.sum()
        df.loc[jump_mask, col] = np.nan

    return df, insane_count, jump_count

###################################
# 3. DETECT INCOMPLETE ROWS (NO DROPPING)
###################################
def count_incomplete_rows(df, logger_type):
    # this doesnt delete rows or anything, it just counts...maybe superfluous
    if logger_type == "TL":
        incomplete = df['t3'].isna()
    else:
        incomplete = df[['t1','t2','t3']].isna().any(axis=1)

    return incomplete.sum()

###################################
# 4. DELETE DUPLICATE TIMESTAMPS
###################################
# these are usually sensor bugs
def remove_duplicate_timestamps(df):
    before = len(df)
    df = df.sort_values('datetime')
    df = df[~df['datetime'].duplicated(keep='first')]
    removed = before - len(df)
    return df, removed

###################################
# 5. FILL SMALL GAPS (<20 rows)
###################################
# fill gaps that dont exceed 20 entries (3 hours)
# this only fills row entries with NaNs, i.e., stuff that got removed earlier or is just there  
def fill_small_gaps(df, limit=20):
    before_nans = df[['t1','t2','t3','SMC']].isna().sum().sum()

    df[['t1','t2','t3','SMC']] = df[['t1','t2','t3','SMC']].interpolate(
        method='linear',
        limit=limit,
        limit_direction='both'
    )

    after_nans = df[['t1','t2','t3','SMC']].isna().sum().sum()

    filled = before_nans - after_nans
    return df, filled

def run_QA_pipeline(df):
    # determine logger type
    logger_type = detect_logger_type(df)
    # First: ensure datetime is parsed consistently
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    # 1. Remove duplicate timestamps BEFORE reindexing
    df, duplicates_removed = remove_duplicate_timestamps(df)
    # 2. Insert missing timestamps
    df, inserted_missing, large_gaps = add_missingentries(df)
    # 3. Remove non-15min timestamps (rare after reindexing, but good safety)
    df, removed_15min = enforce_15min(df)
    # 4. Remove insane values and jumps
    df, insane_removed, jump_removed = remove_insane_and_jumps(df)
    # 5. Count incomplete rows
    incomplete_rows = count_incomplete_rows(df, logger_type)
    # 6. Fill small gaps with linearly interpolated data
    df, gaps_filled = fill_small_gaps(df)

    return df, {
        "logger_type": logger_type,
        "missing_timestamps_inserted": inserted_missing,
        "large_time_gaps": large_gaps,
        "non15min_removed": removed_15min,
        "insane_values_removed": insane_removed,
        "jump_values_removed": jump_removed,
        "incomplete_rows": incomplete_rows,
        "duplicate_timestamps_removed": duplicates_removed,
        "small_gaps_filled": gaps_filled,
    }

## figure script from TOMST
def create_interactive_figure(df, logger_type, filename, cleaned_filename, figure_folder):
    fig = go.Figure()

    # Always plot t3 (air)
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['t3'],
        mode='lines',
        name='T air'
    ))

    # TMS has t1/t2
    if logger_type == 'TMS':
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['t2'],
            mode='lines',
            name='T surface'
        ))
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['t1'],
            mode='lines',
            name='T soil'
        ))

    # Layout with summer shading
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0="2024-05-15",
                x1="2024-09-15",
                y0=0,
                y1=1,
                fillcolor="yellow",
                opacity=0.2,
                line_width=0,
            )
        ],
        annotations=[
            dict(
                x='2024-07-15',
                y=1.05,
                xref='x',
                yref='paper',
                text='Summer 24',
                showarrow=False,
                font=dict(size=14, color="yellow"),
                align="center"
            )
        ],
        title=f"{'Air temperature' if logger_type == 'TL' else 'Soil, surface, air temperature'}, {filename}",
        xaxis_title='Date',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        template='plotly_dark',
        height=700,
        width=1200
    )

    # File naming
    figure_name = f"{cleaned_filename[:-15]}QA_fig.html"
    figure_path = os.path.join(figure_folder, figure_name)

    # Save interactive figure
    fig.write_html(figure_path)

    return figure_path


# FUNCTION TO PROCESS THROUGH A LOOP
def process_folder(source_folder, output_folder, report_folder, figure_folder):

    report_rows = []

    for file in os.listdir(source_folder):
        if not file.lower().endswith(".csv"):
            continue

        in_path = os.path.join(source_folder, file)
        df = pd.read_csv(in_path)

        # Run QA
        df_clean, stats = run_QA_pipeline(df)
        logger_type = stats["logger_type"]

        # Save cleaned CSV
        cleaned_name = file.replace(".csv", "_QA.csv")
        cleaned_path = os.path.join(output_folder, cleaned_name)
        df_clean.to_csv(cleaned_path, index=False)

        # Save figure
        fig_path = create_interactive_figure(
            df_clean,
            logger_type,
            filename=file,
            cleaned_filename=cleaned_name,
            figure_folder=figure_folder
        )

        # Add to report row
        stats["file"] = file
        stats["figure"] = fig_path
        report_rows.append(stats)

        print(f"Processed {file} → {cleaned_name}")
        print(f"Figure saved: {fig_path}")

    # Compile QA report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"QA_report_{timestamp}.csv"
    report_df = pd.DataFrame(report_rows)
    report_path = os.path.join(report_folder, report_name)
    report_df.to_csv(report_path, index=False)

    print("\n====================================")
    print(" QA COMPLETE, CHECK FILES")
    print("====================================")

###################################
# RUN
###################################
process_folder(SOURCE_FOLDER, OUT_FOLDER, REP_FOLDER, FIG_FOLDER)
