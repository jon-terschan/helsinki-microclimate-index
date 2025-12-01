import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
file_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\date_filtered\output\CLF_94200168_2024_09_22_TMS_filtered.csv"
#CLF_94200168_2024_09_22_TMS_filtered
#CLF_94200112_2024_09_23_TMS_filtered
#CLF_94200120_2024_11_05_TMS_filtered
file_name = os.path.basename(file_path)  # Extract filename only
# Load data
df = pd.read_csv(file_path, parse_dates=['datetime'])

# NO 1: RANGE CHECK ######
# plausible temperature ranges for each data
ranges = {
    't1': (-40, 60),
    't2': (-50, 70),
    't3': (-20, 40),
}

# check for range
def in_range(val, min_val, max_val):
    return (val >= min_val) & (val <= max_val)
for col, (min_val, max_val) in ranges.items():
    df[f'{col}_range_flag'] = ~in_range(df[col], min_val, max_val)

# NO 2: JUMP CHECK ###########
# check for jumps bigger than 5 degrees between data
# adaptive jump detection based on rolling standard deviation
# parameters
win = 24              # ~6 hours (25x15 mins)
n_sigmas = 5          # strictness of hampel filter, the hjigher the stricter
min_abs_jump = 5.0    # ignores small jumps in negative degrees for instance in night, °C
reversal_steps = 5    # change must reverse within 5x15 min intervals (30 min) for it to be considered an outlier
# function to define median absolute deviation MAD in the window
def rolling_mad(x):
    m = np.median(x)
    return np.median(np.abs(x - m))
# hampel filter outlier detection
for col in ['t1', 't2', 't3']:
    d = df[col].diff()
    med = d.rolling(win, center=True, min_periods=win//4).median()
    mad = d.rolling(win, center=True, min_periods=win//4).apply(rolling_mad, raw=True)
    thresh = n_sigmas * 1.4826 * mad #threshold for flag (scaled MAD to approximate SD for normal distribution, multiplied by strictness level)

    # large changes 
    large_change = d.abs() > np.maximum(thresh, min_abs_jump) # check if jump is bigger than rolling hampel threshold and at least minimum jump size

    # reversal check: sign flips back within N steps
    # look ahead reversal_step many points to check if sign of the difference flips (change direction reversal)
    reversal = (
        np.sign(d) != np.sign(d.shift(-reversal_steps))
    ) & d.shift(-reversal_steps).abs().gt(np.maximum(thresh.shift(-reversal_steps), min_abs_jump))

    df[f'{col}_jump_flag'] = (large_change & reversal).fillna(False) # combine conditions
    df.at[df.index[0], f'{col}_jump_flag'] = False # turn off first row flag (no points to compare changes)

# i commented this check out because it is too rudimentary to capture soil temperature out of soil periods and will just randomly flag peak data

# check soil airtime relationship at max temp
#delta = 0.5  # tolerance, it will accept + 0.5 degrees, but may not be necessary.
#df['date'] = df['datetime'].dt.date
#daily_max = df.groupby('date').apply(lambda g: g.loc[g['t1'].idxmax()]) # find max air temp rows
#daily_max['consistency_flag'] = ~(daily_max['t3'] <= daily_max['t1'] + delta) # apply consistency checks at those rows
# initialize consistency flag
#df['consistency_flag'] = False
# Assign daily max consistency flag back to original df
#df.loc[daily_max.index, 'consistency_flag'] = daily_max['consistency_flag']
# Combine initial flags except jump flags for now

initial_flags = [f'{col}_range_flag' for col in ranges] #+ ['consistency_flag']
df['fault_flag_initial'] = df[initial_flags].any(axis=1)
# Now adjust jump flags to ignore jumps right after an initial fault
# this is to ensure data isnt flagged as faulty just because it is right before a faulty jump
for col in ['t1', 't2', 't3']:
    jump_flag_col = f'{col}_jump_flag'
    df[jump_flag_col] = df[jump_flag_col] & (~df['fault_flag_initial'].shift(1).fillna(False))
# Final fault flag includes adjusted jump flags
flag_cols = [f'{col}_range_flag' for col in ranges] + [f'{col}_jump_flag' for col in ['t1','t2','t3']]
 #+ ['consistency_flag']
df['fault_flag'] = df[flag_cols].any(axis=1)

df

# create summary csv
outlier_count = int(df['fault_flag'].sum())
# create/append summary CSV
summary_file = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\outlier_detection\outlier_summary.csv"
summary_df = pd.DataFrame([[file_name, outlier_count]], columns=['filename', 'outlier_count'])
summary_df
if os.path.exists(summary_file):
    # append to existing
    existing_df = pd.read_csv(summary_file)
    existing_df = pd.concat([existing_df, summary_df], ignore_index=True)
    existing_df.to_csv(summary_file, index=False)
else:
    # create new
    summary_df.to_csv(summary_file, index=False)
    # Visualization
plt.figure(figsize=(15, 8))
for i, col in enumerate(['t1', 't2', 't3'], 1):
    plt.subplot(3, 1, i)
    plt.plot(df['datetime'], df[col], label=col, color='blue')
    faulty = df[df['fault_flag']]
    plt.scatter(faulty['datetime'], faulty[col], color='red', label='fault_flag', s=20)
    plt.ylabel(f"{col} (°C)")
    plt.legend()
    plt.grid(True)
plt.xlabel('Datetime')
plt.suptitle('temp time series, faulty flag highlighted')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print(f"Total faulty measurements detected: {df['fault_flag'].sum()}")
