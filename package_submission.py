"""Automated Packager for Amazon ML Challenge 2026.
Constructs the final submission zip archive (<team_name>_submission.zip)
strictly adhering to the required directory format.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

# UTF-8 stdout configuration for Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def create_submission_zip(
    team_name: str,
    output_dir: Path = Path("output"),
    code_dir: Path = Path("code/business_entity_resolution"),
    doc_template: Path = Path("Documentation_template.md"),
    zip_output_dir: Path = Path("."),
) -> Path:
    """Builds <team_name>_submission.zip with strict compliance checks."""
    zip_name = f"{team_name}_submission.zip"
    zip_path = zip_output_dir / zip_name

    if zip_path.exists():
        print(f"[*] Removing existing {zip_path}...")
        os.remove(zip_path)

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

    print(f"[*] Packaging submission into: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add output/ folder
        print("  - Adding output/matching_results.tsv")
        zipf.write(matching_tsv, arcname="output/matching_results.tsv")
        print("  - Adding output/candidate_pairs.tsv")
        zipf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # 2. Add code/business_entity_resolution/src/
        src_dir = code_dir / "src"
        if src_dir.exists():
            for root, _, files in os.walk(src_dir):
                for file in files:
                    if file.endswith(".py"):
                        file_path = Path(root) / file
                        rel_path = file_path.relative_to(code_dir.parent.parent)
                        zipf.write(file_path, arcname=str(rel_path).replace("\\", "/"))
                        print(f"  - Adding {str(rel_path).replace('\\', '/')}")

        # 3. Add code/business_entity_resolution/README.md and requirements.txt
        for extra in ["README.md", "requirements.txt", "run_aws.py"]:
            extra_path = code_dir / extra
            if extra_path.exists():
                rel_path = extra_path.relative_to(code_dir.parent.parent)
                zipf.write(extra_path, arcname=str(rel_path).replace("\\", "/"))
                print(f"  - Adding {str(rel_path).replace('\\', '/')}")

        # 4. Add Documentation_template.md to root of zip
        zipf.write(doc_template, arcname="Documentation_template.md")
        print("  - Adding Documentation_template.md")

    print(f"\n[OK] Submission package created successfully: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Package final submission zip for Amazon ML Challenge 2026")
    parser.add_argument("--team-name", type=str, default="Fusion_Force", help="Your official team name")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory containing TSV outputs")
    parser.add_argument("--code-dir", type=str, default="code/business_entity_resolution", help="Directory containing source code")
    parser.add_argument("--doc", type=str, default="Documentation_template.md", help="Path to methodology document")
    args = parser.parse_args()

    create_submission_zip(
        team_name=args.team_name,
        output_dir=Path(args.output_dir),
        code_dir=Path(args.code_dir),
        doc_template=Path(args.doc),
    )


if __name__ == "__main__":
    main()
