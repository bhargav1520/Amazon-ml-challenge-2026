# How to Run Guide: Amazon ML Challenge 2026

This guide provides the exact commands to run testing, training, inference, validation, and submission packaging across **Windows PowerShell**, **Windows CMD**, and **Linux/macOS Bash**.

---

## ⚠️ Important Note on Multi-Line Commands
* On **Windows PowerShell**, line continuation uses the backtick character (`` ` ``), NOT backslash (`\`).
* Alternatively, run commands as a **single line** (recommended to avoid syntax issues).

---

## 1. Run Automated Unit & Integration Tests

Run the full pytest suite (19 tests):

```bash
pytest code/business_entity_resolution/tests/
```

---

## 2. Run Full Training & Inference Pipeline

### 🔹 Single-Line Command (Recommended for PowerShell / CMD / Bash):
```bash
python code/business_entity_resolution/src/pipeline.py --train-dir student_resource/dataset/train --test-dir student_resource/dataset/test --output-dir output
```

### 🔹 Windows PowerShell (Multi-Line with Backticks `` ` ``):
```powershell
python code/business_entity_resolution/src/pipeline.py `
  --train-dir student_resource/dataset/train `
  --test-dir student_resource/dataset/test `
  --output-dir output
```

### 🔹 Linux / macOS / Git Bash (Multi-Line with Backslash `\`):
```bash
python code/business_entity_resolution/src/pipeline.py \
  --train-dir student_resource/dataset/train \
  --test-dir student_resource/dataset/test \
  --output-dir output
```

> **Outputs generated in `output/`:**
> * `output/matching_results.tsv` (Leaderboard upload)
> * `output/candidate_pairs.tsv` (Blocking candidate set)

---

## 3. Fast Prototyping Slice (Quick Dry Run)

To test on a smaller sample (e.g., 5,000 train rows, 2,000 test rows) in ~20 seconds:

```bash
python code/business_entity_resolution/src/pipeline.py --train-dir student_resource/dataset/train --test-dir student_resource/dataset/test --output-dir output --train-limit 5000 --test-limit 2000
```

---

## 4. Run Official Submission Validator

Always run the official validator on the generated TSVs before submitting to the leaderboard:

### 🔹 Single-Line Command:
```bash
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```

### 🔹 Windows PowerShell:
```powershell
python student_resource/utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir student_resource/dataset/test
```

> **Target Output:** `PASS (exit 0)`

---

## 5. Build Final Submission Zip Package

Create the submission archive for the top 100 round (replace `YourTeamName` with your actual team name):

```bash
python package_submission.py --team-name YourTeamName
```

> **Generated Archive:** `YourTeamName_submission.zip`
