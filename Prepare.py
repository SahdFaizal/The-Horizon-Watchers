from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# 1. Correct base directory
# ---------------------------------------------------------
BASE_DIR = Path(r"C:\Sahd 2026 (Class 11-A)\General\The Horizon Watchers")
RAW_DIR = BASE_DIR / "data" / "raw"

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Raw directory:  {RAW_DIR}")

# ---------------------------------------------------------
# 2. Validate directory exists
# ---------------------------------------------------------
if not RAW_DIR.exists():
    print("❌ ERROR: RAW_DIR does not exist. Check your folder path.")
    raise SystemExit()

print("📄 Files inside RAW_DIR:")
for f in RAW_DIR.iterdir():
    print("   -", f.name)


# ---------------------------------------------------------
# 3. File loader (safe for CSV, Excel, or no extension)
# ---------------------------------------------------------
def load_file(file_path):
    """Load CSV or Excel safely based on extension."""
    ext = file_path.suffix.lower()

    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            # Try CSV first, then Excel
            try:
                return pd.read_csv(file_path)
            except Exception:
                return pd.read_excel(file_path)

    except Exception as e:
        print(f"⚠️ Could not read {file_path.name}: {e}")
        return None


# ---------------------------------------------------------
# 4. Generic combiner function
# ---------------------------------------------------------
def combine_files(pattern, output_name):
    """Combine files matching a pattern into a single CSV."""
    print(f"\n🔍 Searching for files: {pattern}")

    files = sorted([
        f for f in RAW_DIR.glob(pattern)
        if f.name not in ["modis.csv", "viirs.csv"]
    ])

    if not files:
        print(f"⚠️ No files found for pattern: {pattern}")
        return

    print(f"📦 Found {len(files)} file(s). Combining...")

    dfs = []
    for f in files:
        df = load_file(f)
        if df is not None:
            dfs.append(df)
        else:
            print(f"⚠️ Skipping unreadable file: {f.name}")

    if not dfs:
        print(f"❌ No readable data found for {pattern}. Nothing to combine.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    output_path = RAW_DIR / output_name
    combined.to_csv(output_path, index=False)

    print(f"✅ Success! Saved combined file to: {output_path}")


# ---------------------------------------------------------
# 5. Combine MODIS and VIIRS
# ---------------------------------------------------------
combine_files("MODIS_SP_*", "modis.csv")
combine_files("VIIRS_NOAA20_SP_*", "viirs.csv")
