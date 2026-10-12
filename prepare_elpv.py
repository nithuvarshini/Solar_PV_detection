"""
prepare_elpv.py
───────────────────────────────────────────────────────────────────────────────
Downloads and prepares the ELPV (Electroluminescence Photovoltaic) dataset
for the Solar PV Defect Detection project.

USAGE
─────
Option A – Automatic Kaggle download (recommended):
    1. Install kaggle CLI:   pip install kaggle
    2. Put your API key at:  C:/Users/<you>/.kaggle/kaggle.json
       (Download from https://www.kaggle.com/settings → "Create New Token")
    3. Run:  python prepare_elpv.py

Option B – Manual download:
    1. Go to: https://www.kaggle.com/datasets/lsjsj92/kee-elpv-dataset
    2. Download and extract the ZIP.
    3. Pass the extracted folder path:
       python prepare_elpv.py --zip path/to/archive.zip

After running this script, edit config/config.yaml and set:
    dataset:
      dataset_mode: "elpv"

Then run:  python run_pipeline.py
───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import shutil
import argparse
import zipfile
import numpy as np
import pandas as pd
from PIL import Image

# ─── ELPV defect probability → 4-class label mapping ──────────────────────────
# The ELPV dataset labels each cell with a defect probability in {0, 1/3, 2/3, 1}
# and a type (mono / poly).  We map those probabilities to our 4 classes:
#   prob = 0.00  → class 0: Healthy
#   prob = 0.33  → class 1: Micro-Crack  (light defect)
#   prob = 0.67  → class 2: Finger Interruption (moderate defect)
#   prob = 1.00  → class 3: Delamination / Severe

PROB_TO_CLASS = {0.0: 0, 1/3: 1, 2/3: 2, 1.0: 3}
CLASS_NAMES   = ["Healthy", "Micro-Crack", "Finger Interruption", "Delamination"]

OUTPUT_DIR = os.path.join("data", "elpv")
IMG_OUT    = os.path.join(OUTPUT_DIR, "images")


def prob_to_label(prob: float) -> int:
    """Round defect probability to nearest ELPV bin and return class id."""
    bins   = [0.0, 1/3, 2/3, 1.0]
    dists  = [abs(prob - b) for b in bins]
    return PROB_TO_CLASS[bins[int(np.argmin(dists))]]


def synthetic_telemetry(label: int) -> dict:
    """Generate plausible panel telemetry correlated with defect severity."""
    params = {
        0: dict(voc=(0.62, 0.65), isc=(9.0, 9.5), ff=(78.0, 82.0),
                rs=(0.010, 0.020), rsh=(400, 500), td=(0.5, 2.0),
                el_m=(160, 180), el_s=(15, 25)),
        1: dict(voc=(0.57, 0.61), isc=(8.7, 9.2), ff=(71.0, 76.0),
                rs=(0.025, 0.040), rsh=(250, 350), td=(2.5, 5.0),
                el_m=(140, 160), el_s=(30, 45)),
        2: dict(voc=(0.59, 0.63), isc=(8.0, 8.6), ff=(73.0, 77.0),
                rs=(0.035, 0.055), rsh=(300, 420), td=(3.0, 6.0),
                el_m=(145, 165), el_s=(25, 38)),
        3: dict(voc=(0.51, 0.56), isc=(7.5, 8.3), ff=(62.0, 70.0),
                rs=(0.050, 0.080), rsh=(100, 220), td=(6.5, 12.0),
                el_m=(115, 140), el_s=(45, 65)),
    }
    p = params[label]
    return {
        'Voc':               round(np.random.uniform(*p['voc']), 4),
        'Isc':               round(np.random.uniform(*p['isc']), 4),
        'Fill_Factor':       round(np.random.uniform(*p['ff']),  2),
        'Series_Resistance': round(np.random.uniform(*p['rs']),  4),
        'Shunt_Resistance':  round(np.random.uniform(*p['rsh']), 2),
        'Temperature_Delta': round(np.random.uniform(*p['td']),  2),
        'EL_Intensity_Mean': round(np.random.uniform(*p['el_m']), 2),
        'EL_Intensity_Std':  round(np.random.uniform(*p['el_s']), 2),
    }


def extract_zip(zip_path: str, extract_to: str):
    print(f"Extracting {zip_path} → {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print("Extraction complete.")


def find_elpv_root(search_dir: str) -> str:
    """Walk extracted folder to find the directory containing labels.csv."""
    for root, dirs, files in os.walk(search_dir):
        if "labels.csv" in files or "elpv_labels.csv" in files:
            return root
    # Also try looking for a txt labels file
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if "label" in f.lower() and f.endswith((".csv", ".txt")):
                return root
    return search_dir


def build_elpv_csv(elpv_root: str):
    """Parse ELPV labels, copy images, and write elpv_dataset.csv."""
    os.makedirs(IMG_OUT, exist_ok=True)

    # Find label file
    label_file = None
    for fname in ["labels.csv", "elpv_labels.csv", "labels.txt"]:
        candidate = os.path.join(elpv_root, fname)
        if os.path.exists(candidate):
            label_file = candidate
            break

    if label_file is None:
        raise FileNotFoundError(
            f"Could not find labels.csv inside {elpv_root}.\n"
            "Please make sure you extracted the correct ELPV dataset ZIP."
        )

    print(f"Reading labels from: {label_file}")

    # Official ELPV labels.csv is whitespace-separated:
    # filename probability type
    ldf = pd.read_csv(
        label_file,
        sep=r"\s+",
        header=None,
        names=["filename", "probability", "type"]
    )

    # Normalise column names
    ldf.columns = [c.strip().lower() for c in ldf.columns]
    ldf = ldf.rename(columns={"filename": "filename", "probability": "probability"})

    if "filename" not in ldf.columns:
        ldf.columns = ["filename", "probability"] + list(ldf.columns[2:])

    print(f"Found {len(ldf)} labelled entries.")
    records = []

    for idx, row in ldf.iterrows():
        orig_name = str(row["filename"]).strip()
        # Try locating image relative to elpv_root
        for search in [elpv_root,
                       os.path.join(elpv_root, "images"),
                       os.path.join(elpv_root, "data")]:
            src = os.path.join(search, orig_name)
            if os.path.exists(src):
                break
        else:
            # Last resort: walk the whole extraction dir
            found = None
            base = os.path.basename(orig_name)
            for rt, _, files in os.walk(elpv_root):
                if base in files:
                    found = os.path.join(rt, base)
                    break
            src = found if found else src

        if not os.path.exists(src):
            continue  # Skip missing images

        prob  = float(row.get("probability", 0.0))
        label = prob_to_label(prob)

        out_name = f"elpv_{idx:04d}_cls{label}.png"
        dst_path = os.path.join(IMG_OUT, out_name)

        # Convert to RGB PNG and copy
        try:
            pil = Image.open(src).convert("RGB")
            pil.save(dst_path)
        except Exception as e:
            print(f"  [WARN] Skipping {src}: {e}")
            continue

        tel = synthetic_telemetry(label)
        rec = {
            "sample_id":    idx,
            "image_name":   out_name,
            "label":        label,
            "defect_class": CLASS_NAMES[label],
            "orig_file":    orig_name,
            "defect_prob":  prob,
        }
        rec.update(tel)
        records.append(rec)

    df = pd.DataFrame(records)
    csv_out = os.path.join(OUTPUT_DIR, "elpv_dataset.csv")
    df.to_csv(csv_out, index=False)

    print(f"\nDataset prepared successfully!")
    print(f"  Images   → {IMG_OUT}  ({len(df)} images)")
    print(f"  CSV      → {csv_out}")
    print(f"\nClass distribution:")
    print(df.groupby(["label","defect_class"]).size().to_string())
    print(f"\nNext step: edit config/config.yaml and set  dataset_mode: \"elpv\"")
    print("Then run:  python run_pipeline.py")


def download_from_kaggle():
    """Download ELPV dataset using kaggle CLI."""
    try:
        import kaggle  # noqa
    except ImportError:
        print("kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    os.makedirs("data/tmp_kaggle", exist_ok=True)
    print("Downloading ELPV dataset from Kaggle (lsjsj92/kee-elpv-dataset)...")
    os.system("kaggle datasets download -d lsjsj92/kee-elpv-dataset -p data/tmp_kaggle --unzip")
    return "data/tmp_kaggle"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ELPV solar cell EL dataset for training."
    )
    parser.add_argument(
        "--zip", type=str, default=None,
        help="Path to manually downloaded ELPV ZIP archive."
    )
    parser.add_argument(
        "--extracted", type=str, default=None,
        help="Path to an already-extracted ELPV folder (containing labels.csv)."
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.extracted:
        elpv_root = find_elpv_root(args.extracted)
    elif args.zip:
        extract_dir = os.path.join("data", "elpv_extracted_tmp")
        extract_zip(args.zip, extract_dir)
        elpv_root = find_elpv_root(extract_dir)
    else:
        # Try automatic Kaggle download
        tmp_dir = download_from_kaggle()
        elpv_root = find_elpv_root(tmp_dir)

    build_elpv_csv(elpv_root)


if __name__ == "__main__":
    main()
