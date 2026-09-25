"""Automated Packager for Amazon ML Challenge 2026.
Constructs the final submission zip archive (<team_name>_submission.zip)
strictly adhering to the required directory format.
"""

import argparse
import os
import zipfile
from pathlib import Path


def create_submission_zip(
    team_name: str,
    output_dir: Path = Path("output"),
    code_dir: Path = Path("code/business_entity_resolution"),
    doc_template: Path = Path("student_resource/Documentation_template.md"),
    zip_output_dir: Path = Path("."),
) -> Path:
    """Builds <team_name>_submission.zip with strict compliance checks."""
    zip_name = f"{team_name}_submission.zip"
    zip_path = zip_output_dir / zip_name

    # Verification of required files
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"

    if not matching_tsv.exists():
        raise FileNotFoundError(f"Missing required file: {matching_tsv}")
    if not candidate_tsv.exists():
        raise FileNotFoundError(f"Missing required file: {candidate_tsv}")
    if not code_dir.exists():
        raise FileNotFoundError(f"Missing code directory: {code_dir}")
    if not doc_template.exists():
        raise FileNotFoundError(f"Missing documentation template: {doc_template}")

    print(f"📦 Packaging submission into: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add output/ folder
        zipf.write(matching_tsv, arcname="output/matching_results.tsv")
        zipf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # 2. Add code/business_entity_resolution/ (src/, README.md, requirements.txt)
        for root, dirs, files in os.walk(code_dir):
            # Skip test and cache directories
            dirs[:] = [d for d in dirs if d not in {"__pycache__", ".pytest_cache", "tests"}]
            for file in files:
                if file.endswith((".pyc", ".DS_Store")):
                    continue
                file_path = Path(root) / file
                rel_path = file_path.relative_to(code_dir.parent.parent)
                zipf.write(file_path, arcname=str(rel_path).replace("\\", "/"))

        # 3. Add Documentation_template.md to root of zip
        zipf.write(doc_template, arcname="Documentation_template.md")

    print(f"✅ Submission package created successfully: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Package final submission zip for Amazon ML Challenge 2026")
    parser.add_argument("--team-name", type=str, required=True, help="Your official team name")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory containing TSV outputs")
    parser.add_argument("--code-dir", type=str, default="code/business_entity_resolution", help="Directory containing source code")
    parser.add_argument("--doc", type=str, default="student_resource/Documentation_template.md", help="Path to methodology document")
    args = parser.parse_args()

    create_submission_zip(
        team_name=args.team_name,
        output_dir=Path(args.output_dir),
        code_dir=Path(args.code_dir),
        doc_template=Path(args.doc),
    )


if __name__ == "__main__":
    main()
