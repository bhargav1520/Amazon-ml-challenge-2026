"""Reliable Google Drive dataset downloader for Lightning AI Studio."""

import os
import sys
import zipfile
import shutil
from pathlib import Path
import subprocess

FOLDER_URL = "https://drive.google.com/drive/folders/1d7aK6mkeTaJ4Dns0KWddOZ1RO2JvM6Z3"

def main():
    print("[*] Starting Dataset Download from Google Drive...")
    download_dir = Path("dataset_download")
    download_dir.mkdir(exist_ok=True)
    
    # Run gdown to fetch the folder
    subprocess.run(["gdown", "--folder", FOLDER_URL, "-O", str(download_dir), "--remaining-ok"])
    
    # Extract any zip files found
    for zip_path in list(download_dir.rglob("*.zip")) + list(Path(".").glob("*.zip")):
        if "Fusion_Force" not in zip_path.name:
            print(f"[*] Extracting archive: {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            
    # Locate train and test folders
    target_dataset = Path("dataset")
    target_dataset.mkdir(exist_ok=True)
    
    train_dirs = [d for d in download_dir.rglob("train") if (d / "train_ground_truth.tsv").exists()]
    test_dirs = [d for d in download_dir.rglob("test") if (d / "test_source1.tsv").exists()]
    
    if train_dirs:
        print(f"[*] Found training data at: {train_dirs[0]}")
        if (target_dataset / "train").exists():
            shutil.rmtree(target_dataset / "train")
        shutil.copytree(train_dirs[0], target_dataset / "train")
    else:
        print("[!] Warning: train folder not found in download. Checking fallback...")
        
    if test_dirs:
        print(f"[*] Found test data at: {test_dirs[0]}")
        if (target_dataset / "test").exists():
            shutil.rmtree(target_dataset / "test")
        shutil.copytree(test_dirs[0], target_dataset / "test")
    else:
        print("[!] Warning: test folder not found in download. Checking fallback...")
        
    print("\n[✓] Final dataset status:")
    if (target_dataset / "train").exists():
        for f in sorted((target_dataset / "train").glob("*.tsv")):
            print(f"  - train/{f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
    if (target_dataset / "test").exists():
        for f in sorted((target_dataset / "test").glob("*.tsv")):
            print(f"  - test/{f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    main()
