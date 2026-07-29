import glob
import os
import sys

import pandas as pd

DATA_DIR = "/Users/mghrear/data/HPS_data/2021_v9_pass5_TC_processed/"

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pk")))
print(f"{len(files)} files found")

failed = []
for i, f in enumerate(files):
    print(f"[{i+1}/{len(files)}] {os.path.basename(f)}", end="\r")
    try:
        df = pd.read_pickle(f)
        df.columns = pd.Index([str(c) for c in df.columns], dtype=object)
        tmp_path = f + ".tmp"
        df.to_pickle(tmp_path)
        os.replace(tmp_path, f)  # atomic on same filesystem
    except Exception as e:
        failed.append((f, str(e)))
        print(f"\nFAILED: {f}: {e}")

print(f"\nDone. {len(files) - len(failed)} succeeded, {len(failed)} failed.")
if failed:
    for f, e in failed:
        print(f"  {f}: {e}")
    sys.exit(1)
