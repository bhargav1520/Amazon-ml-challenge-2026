#!/usr/bin/env python3
"""Submission Validator for Amazon ML Challenge 2026."""

import argparse
import os
import sys

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="Validate submission TSV files")
    parser.add_argument("--matching", type=str, required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=str, default="", help="Path to candidate_pairs.tsv (optional)")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test dataset directory")
    args = parser.parse_args()

    print("=" * 60)
    print("ML Challenge 2026 — Submission Validator")
    print("=" * 60)

    # 1. Check matching file exists
    if not os.path.isfile(args.matching):
        print(f"FAIL: Matching file not found: {args.matching}")
        sys.exit(1)

    # 2. Check header & rows
    s1_ids = []
    matched_rows = 0
    empty_rows = 0
    duplicate_s1 = 0
    seen_s1 = set()

    with open(args.matching, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(DELIM)
        if header != MATCHING_HEADER:
            print(f"FAIL: Invalid matching file header. Expected {MATCHING_HEADER}, got {header}")
            sys.exit(1)

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split(DELIM)
            if len(parts) < 2:
                # empty matched_entity_ids column
                s1_id = parts[0].strip()
                match_str = ""
            else:
                s1_id = parts[0].strip()
                match_str = parts[1].strip()

            if s1_id in seen_s1:
                duplicate_s1 += 1
            seen_s1.add(s1_id)
            s1_ids.append(s1_id)

            if match_str:
                matched_rows += 1
            else:
                empty_rows += 1

    total_s1 = len(s1_ids)
    print(f"[*] matching_results file: {total_s1:,} rows total")
    print(f"    - Matched entities:   {matched_rows:,} ({matched_rows/total_s1*100:.2f}%)")
    print(f"    - Singletons (empty): {empty_rows:,} ({empty_rows/total_s1*100:.2f}%)")

    if total_s1 != 1732544:
        print(f"WARNING: Expected 1,732,544 rows, found {total_s1:,}")
    if duplicate_s1 > 0:
        print(f"FAIL: Found {duplicate_s1} duplicate source1_entity_id entries!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PASS — no blocking issues found. Safe to submit!")
    print("=" * 60)


if __name__ == "__main__":
    main()
