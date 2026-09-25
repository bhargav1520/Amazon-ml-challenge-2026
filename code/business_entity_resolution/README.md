# Business Entity Resolution Pipeline

This folder contains the complete, self-contained, runnable Machine Learning pipeline for the Amazon ML Challenge 2026.

## 🛠️ Requirements & Setup

Install dependencies using:
```bash
pip install -r requirements.txt
```

## 🚀 End-to-End Pipeline Execution

Run the complete pipeline from the project root:
```bash
python -m src.pipeline \
  --train-dir student_resource/dataset/train \
  --test-dir student_resource/dataset/test \
  --output-dir output
```

### Prototyping / Fast Slice Flags:
To run a fast sanity check on a small subset (e.g. 10,000 rows):
```bash
python -m src.pipeline \
  --train-dir student_resource/dataset/train \
  --test-dir student_resource/dataset/test \
  --output-dir output \
  --train-limit 10000 \
  --test-limit 5000
```

## 🧪 Running Automated Tests

Run the full pytest suite:
```bash
pytest code/business_entity_resolution/tests/
```

## 📋 Pre-Submission Validation

Validate generated TSV outputs before portal submission:
```bash
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```
