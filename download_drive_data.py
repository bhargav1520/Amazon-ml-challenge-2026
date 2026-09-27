"""Auto-downloader for Google Drive dataset folder in Lightning AI."""

import os
import zipfile
from pathlib import Path
import subprocess

FOLDER_URL = "https://drive.google.com/drive/folders/1d7aK6mkeTaJ4Dns0KWddOZ1RO2JvM6Z3"

def main():
    print("[*] Downloading dataset from Google Drive folder...")
    download_dir = Path("dataset_download")
    download_dir.mkdir(exist_ok=True)
    
    # Run gdown
    subprocess.run(["gdown", "--folder", FOLDER_URL, "-O", str(download_dir), "--remaining-ok"])
    
    # Extract any zip files found
    for zip_path in download_dir.rglob("*.zip"):
        print(f"[*] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
            
    # Locate train and test folders
    target_dataset = Path("dataset")
    target_dataset.mkdir(exist_ok=True)
    
    train_dirs = list(download_dir.rglob("train"))
    test_dirs = list(download_dir.rglob("test"))
    
    if train_dirs and (train_dirs[0] / "train_ground_truth.tsv").exists():
        import shutil
        print(f"[*] Found training data at {train_dirs[0]}")
        if (target_dataset / "train").exists():
            shutil.rmtree(target_dataset / "train")
        shutil.copytree(train_dirs[0], target_dataset / "train")
        
    if test_dirs:
        import shutil
        print(f"[*] Found test data at {test_dirs[0]}")
        if (target_dataset / "test").exists():
            shutil.rmtree(target_dataset / "test")
        shutil.copytree(test_dirs[0], target_dataset / "test")
        
    print("[✓] Dataset successfully positioned in dataset/ :")
    if (target_dataset / "train").exists():
        for f in (target_dataset / "train").glob("*.tsv"):
            print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    main()
