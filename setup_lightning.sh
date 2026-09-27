#!/bin/bash
# ==============================================================================
# 1-Click Automated Setup & Runner for Lightning AI Studio
# Google Drive Folder: 1d7aK6mkeTaJ4Dns0KWddOZ1RO2JvM6Z3
# ==============================================================================

set -e

echo "=================================================================="
echo "🚀 [LIGHTNING AI] STARTING 1-CLICK AUTOMATED SETUP & PIPELINE"
echo "=================================================================="

# 1. Install gdown and high-performance ML dependencies
echo "[1/4] Installing Python libraries..."
pip install --upgrade pip -q
pip install -q gdown lightgbm catboost rapidfuzz joblib pyarrow fastparquet pandas numpy scikit-learn pytest

# 2. Download dataset from Google Drive folder if not already present
echo "[2/4] Downloading dataset from Google Drive..."
mkdir -p dataset_download
cd dataset_download

# Download entire Drive folder or zip
gdown --folder "https://drive.google.com/drive/folders/1d7aK6mkeTaJ4Dns0KWddOZ1RO2JvM6Z3" --remaining-ok || true

# Look for zip files or unzipped folders
ZIP_FILES=$(find . -name "*.zip")
for z in $ZIP_FILES; do
    echo "Extracting $z..."
    unzip -q -o "$z" -d extracted/
done

cd ..

# Place train and test directories into dataset/
mkdir -p dataset
if [ -d "dataset_download/extracted/train" ]; then
    cp -r dataset_download/extracted/train dataset/
    cp -r dataset_download/extracted/test dataset/
elif [ -d "dataset_download/train" ]; then
    cp -r dataset_download/train dataset/
    cp -r dataset_download/test dataset/
elif [ -d "dataset_download/*/train" ]; then
    PARENT_DIR=$(find dataset_download -type d -name "train" | head -n 1 | sed 's#/train##')
    cp -r "$PARENT_DIR/train" dataset/
    cp -r "$PARENT_DIR/test" dataset/
fi

echo "[✓] Dataset ready. Checking contents:"
ls -lh dataset/train/ || true

# 3. Execute the full high-capacity 4-stage pipeline
echo "[3/4] Running High-Capacity Pipeline..."
cd code/business_entity_resolution
python3 run_aws.py

# 4. Package final submission zip
echo "[4/4] Creating Final Submission Package..."
cd ../..
python3 package_submission.py --team-name "Fusion_Force"

echo "=================================================================="
echo "🎉 PIPELINE & PACKAGING FULLY COMPLETE!"
echo "Files ready:"
echo "  1. code/business_entity_resolution/output/matching_results.tsv (Upload to Leaderboard)"
echo "  2. Fusion_Force_submission.zip (Final Zip Package)"
echo "=================================================================="
