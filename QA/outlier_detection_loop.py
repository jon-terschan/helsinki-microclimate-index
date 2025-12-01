import os, glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------- Config ----------
RANGES = {'t1': (-40, 60), 't2': (-50, 70), 't3': (-20, 40)}

# Global jump parameters (applied to all sensors)
JUMP_PARAMS = dict(win=24, n_sigmas=5, min_abs_jump=5.0, reversal_steps=5)

SUMMARY_CSV = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\outlier_detection\outlier_summary.csv"
INPUT_GLOB = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\outlier_detection\source\*_TMS_filtered.csv"
FIGURES_DIR = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\outlier_detection\figures"

os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------- Core logic ----------
def _rolling_mad(x):
    m = np.median(x)
    return np.median(np.abs(x - m))


def detect_outliers(df: pd.DataFrame, ranges: dict, jump_params: dict) -> pd.DataFrame:
    """Adds *_range_flag, *_jump_flag, and fault_flag columns to df and returns df."""
    df = df.copy()
    if 'datetime' in df.columns:
        df = df.sort_values('datetime')
    df = df.reset_index(drop=True)

    # Range flags
    for col, (lo, hi) in ranges.items():
        if col in df:
            df[f'{col}_range_flag'] = ~((df[col] >= lo) & (df[col] <= hi))
        else:
            df[f'{col}_range_flag'] = False

    # Robust jump flags
    for col in [c for c in ranges if c in df]:
        p = jump_params
        d = df[col].diff()
        med = d.rolling(p['win'], center=True, min_periods=max(4, p['win'] // 4)).median()
        mad = d.rolling(p['win'], center=True, min_periods=max(4, p['win'] // 4)).apply(_rolling_mad, raw=True)
        thresh = p['n_sigmas'] * 1.4826 * mad

        large_change = d.abs() > np.maximum(thresh, p['min_abs_jump'])
        rev = (
            np.sign(d) != np.sign(d.shift(-p['reversal_steps']))
        ) & (
            d.shift(-p['reversal_steps']).abs() > np.maximum(thresh.shift(-p['reversal_steps']), p['min_abs_jump'])
        )

        df[f'{col}_jump_flag'] = (large_change & rev).fillna(False)
        if len(df) > 0:
            df.at[df.index[0], f'{col}_jump_flag'] = False

    # Combine flags
    flag_cols = [f'{c}_range_flag' for c in ranges if f'{c}_range_flag' in df] + \
                [f'{c}_jump_flag'  for c in ranges if f'{c}_jump_flag'  in df]
    df['fault_flag'] = df[flag_cols].any(axis=1) if flag_cols else False
    return df


def summarize_file(file_path: str) -> tuple[str, int, pd.DataFrame]:
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df = detect_outliers(df, RANGES, JUMP_PARAMS)
    return os.path.basename(file_path), int(df['fault_flag'].sum()), df


def plot_timeseries(df: pd.DataFrame, file_name: str):
    fig = go.Figure()
    for col in [c for c in RANGES if c in df]:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df[col],
                                 mode='lines', name=col, line=dict(width=2)))
        # overlay faulty points
        faulty = df[df['fault_flag']]
        fig.add_trace(go.Scatter(x=faulty['datetime'], y=faulty[col],
                                 mode='markers', name=f"{col} fault",
                                 marker=dict(color='red', size=6)))
    fig.update_layout(title=f"Time Series: {file_name}",
                      xaxis_title='Datetime', yaxis_title='Temperature (°C)',
                      template='plotly_white')
    fig.write_html(os.path.join(FIGURES_DIR, f"{file_name}.html"))
    fig.write_image(os.path.join(FIGURES_DIR, f"{file_name}.png"))


# ---------- Batch run ----------
def run_batch(input_glob: str, summary_csv: str):
    files = sorted(glob.glob(input_glob))
    rows = []
    for fp in files:
        fname, n_out, df = summarize_file(fp)
        rows.append((fname, n_out))
        print(f"Processed {fname}: {n_out} outliers")
        plot_timeseries(df, fname.replace(".csv", ""))

    result = pd.DataFrame(rows, columns=['filename', 'outlier_count'])
    if os.path.exists(summary_csv):
        existing = pd.read_csv(summary_csv)
        out = pd.concat([existing, result], ignore_index=True)
    else:
        out = result
    out.to_csv(summary_csv, index=False)
    return out


if __name__ == "__main__":
    run_batch(INPUT_GLOB, SUMMARY_CSV)


    # delte summary table if it exists before running
    # delete flagged points from table and export as cleaned df
    # double check thresholds 
    # one of the sensors is effin crooked
    # test simple range check...
    # then go into out of soil period checker
    

