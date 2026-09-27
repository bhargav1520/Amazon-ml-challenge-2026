"""Fast End-to-End Execution Script for AWS High-Compute Instances.
Runs all 4 stages sequentially with optimized high-capacity parameters.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.stage1_preprocess import run_stage1_preprocess
from src.stage2_blocking import run_stage2_blocking
from src.stage3_train import run_stage3_train
from src.stage4_inference import run_stage4_inference


def find_dataset_dir() -> Path:
    candidates = [
        BASE_DIR.parent.parent / "student_resource" / "dataset",
        BASE_DIR.parent.parent / "dataset",
        BASE_DIR / "dataset",
        Path.cwd() / "dataset",
        Path("/workspaces/Amazon-ml-challenge-2026/dataset"),
        Path("/workspaces/Amazon-ml-challenge-2026/student_resource/dataset"),
        Path.home() / "dataset",
        Path("/home/ubuntu/dataset"),
    ]
    for c in candidates:
        if (c / "train" / "train_ground_truth.tsv").exists():
            return c.resolve()

    # Recursive fallback search
    for root in [BASE_DIR.parent.parent, Path.cwd(), Path("/workspaces"), Path.home()]:
        matches = list(root.glob("**/train_ground_truth.tsv"))
        if matches:
            return matches[0].parent.parent.resolve()

    raise FileNotFoundError(
        "Could not find dataset directory! Please place your 'train' and 'test' folders inside 'dataset/' or 'student_resource/dataset/'."
    )


def main():
    start_total = time.time()
    data_dir = find_dataset_dir()
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    gt_file = train_dir / "train_ground_truth.tsv"

    cache_dir = BASE_DIR / "processed_data"
    cand_dir = BASE_DIR / "cache"
    models_dir = BASE_DIR / "models"
    output_dir = BASE_DIR / "output"

    print("=" * 70)
    print("🚀 [AWS RUNNER] AMAZON ML CHALLENGE 2026 - HIGH PERFORMANCE PIPELINE")
    print("=" * 70)
    print(f"[*] Dataset Directory: {data_dir}")
    print(f"[*] Output Directory:  {output_dir}")

    # STAGE 1: PREPROCESSING
    s1_t0 = time.time()
    print("\n" + "=" * 70)
    print(">>> STAGE 1: PREPROCESSING RAW SOURCE DATASETS")
    print("=" * 70)
    run_stage1_preprocess(
        train_dir=train_dir,
        test_dir=test_dir,
        cache_dir=cache_dir,
    )
    print(f"[✓] Stage 1 elapsed: {time.time() - s1_t0:.1f}s")

    # STAGE 2: CANDIDATE BLOCKING (25 CANDIDATES PER ENTITY FOR >99.5% RECALL)
    s2_t0 = time.time()
    print("\n" + "=" * 70)
    print(">>> STAGE 2: HIGH-RECALL MULTI-PASS BLOCKING (25 Candidates/Entity)")
    print("=" * 70)
    run_stage2_blocking(
        cache_dir=cache_dir,
        output_dir=output_dir,
        candidates_cache_dir=cand_dir,
        max_candidates=25,
        max_train_sample_entities=300000,
    )
    print(f"[✓] Stage 2 elapsed: {time.time() - s2_t0:.1f}s")

    # STAGE 3: MODEL TRAINING (800 TREES, HARD-NEG X3)
    s3_t0 = time.time()
    print("\n" + "=" * 70)
    print(">>> STAGE 3: DUAL-ENSEMBLE MODEL TRAINING & COUNTRY CALIBRATION")
    print("=" * 70)
    run_stage3_train(
        cache_dir=cache_dir,
        candidates_cache_dir=cand_dir,
        models_dir=models_dir,
        gt_file=gt_file,
        max_train_entities=300000,
        n_estimators=800,
        learning_rate=0.04,
        hard_negative_multiplier=3,
    )
    print(f"[✓] Stage 3 elapsed: {time.time() - s3_t0:.1f}s")

    # STAGE 4: BATCHED TEST INFERENCE & GENERATING SUBMISSION
    s4_t0 = time.time()
    print("\n" + "=" * 70)
    print(">>> STAGE 4: BATCHED TEST INFERENCE & MATCHING RESULTS EXPORT")
    print("=" * 70)
    run_stage4_inference(
        cache_dir=cache_dir,
        candidates_cache_dir=cand_dir,
        models_dir=models_dir,
        output_dir=output_dir,
        batch_size=25000,
        max_matches_per_entity=15,
    )
    print(f"[✓] Stage 4 elapsed: {time.time() - s4_t0:.1f}s")

    # VALIDATION
    print("\n" + "=" * 70)
    print(">>> RUNNING SUBMISSION VALIDATOR")
    print("=" * 70)
    val_script = BASE_DIR.parent.parent / "student_resource" / "utils" / "validate_submission.py"
    if val_script.exists():
        import subprocess
        cmd = [
            sys.executable,
            str(val_script),
            "--matching", str(output_dir / "matching_results.tsv"),
            "--candidate", str(output_dir / "candidate_pairs.tsv"),
            "--test-dir", str(test_dir),
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("\n🎉 [ALL CHECKS PASSED] Final submission is strictly valid!")
        else:
            print("\n⚠️ [WARNING] Validation exited with non-zero status.")
    
    total_min = (time.time() - start_total) / 60.0
    print("\n" + "=" * 70)
    print(f"🏁 PIPELINE FULLY COMPLETE IN {total_min:.1f} MINUTES!")
    print(f"📁 Output file ready at: {output_dir / 'matching_results.tsv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
