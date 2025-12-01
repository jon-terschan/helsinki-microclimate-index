import pandas as pd
import os
import glob
import json

# --- CONFIG ---
INPUT_FOLDER = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\processed\3_gapfiller\source"
DATETIME_COLUMN = "datetime"   # change if needed
OUTPUT_REPORT = "duplicate_summary.json"
MAX_PRINT = 20                 # how many duplicate rows to print per file
# -----------------------------------------------

results = {}

for path in glob.glob(os.path.join(INPUT_FOLDER, "*.csv")):
    file = os.path.basename(path)
    print("\n=== Checking:", file, "===")

    try:
        df = pd.read_csv(path)

        # Parse datetime column
        df['dt'] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce')

        # Duplicate mask
        dup_mask = df['dt'].duplicated(keep=False)
        duplicate_rows = df[dup_mask].sort_values("dt")
        dup_count = dup_mask.sum()
        uniq_count = df['dt'].nunique()

        # Find which timestamps are duplicated
        duplicated_times = df['dt'][df['dt'].duplicated()].unique()

        # Store summary results
        results[file] = {
            "total_rows": int(len(df)),
            "duplicate_rows": int(dup_count),
            "unique_timestamps": int(uniq_count),
            "duplicated_timestamps": duplicated_times.astype(str).tolist()
        }

        # Save duplicate rows to file
        if dup_count > 0:
            out_path = os.path.join(INPUT_FOLDER, file.replace(".csv", "_duplicates.csv"))
            duplicate_rows.to_csv(out_path, index=False)

            print(f"→ Found {dup_count} duplicated rows")
            print(f"→ Saved to {out_path}\n")

            # Print sample of duplicate timestamps
            print("Duplicated timestamps:")
            for ts in duplicated_times[:20]:
                print("   ", ts)
            if len(duplicated_times) > 20:
                print("   ...")

            # Print sample duplicate rows
            print("\nSample duplicate rows:")
            print(duplicate_rows.head(MAX_PRINT))

        else:
            print("→ No duplicates found.")

    except Exception as e:
        results[file] = {"error": str(e)}
        print(f"ERROR in {file}: {e}")

# Save full summary file
with open(os.path.join(INPUT_FOLDER, OUTPUT_REPORT), "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Summary saved as", OUTPUT_REPORT)


# lets turn this into a script that
# - deltes non 15min timestamps alltogether
# - remove insane values and step jumps
# - check NaN or partially filled rows
# - deletes duplicated timestamps
# - detects smaller gaps in data and fills them accordingly, maybe just linear interpolation is enough
