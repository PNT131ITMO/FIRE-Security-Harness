from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

FEVER_REPO = "fever/fever"
FEVER_REVISION = "1c55aad95efe327fc6e71a5893d6040d93a8f98c"
FEVER_FILES = {
    "train": "v1.0/fever-train.parquet",
    "paper_dev": "v1.0/fever-paper_dev.parquet",
    "paper_test": "v1.0/fever-paper_test.parquet",
}

LABEL_MAP = {
    "SUPPORTS": "True",
    "REFUTES": "False",
}

def download_fever_files() -> dict[str, str]:

    local_files = {}

    print("Downloading FEVER parquet files...")

    for split_name, filename in FEVER_FILES.items():

        print(f"  {split_name}: {filename}")

        path = hf_hub_download(
            repo_id=FEVER_REPO,
            repo_type="dataset",
            filename=filename,
            revision=FEVER_REVISION,
        )

        local_files[split_name] = path

    return local_files

def load_fever_dataset():

    local_files = download_fever_files()

    print("\nLoading FEVER parquet files...")

    dataset = load_dataset(
        "parquet",
        data_files=local_files,
    )

    return dataset

def prepare_split(dataset, split_name: str):

    claims_by_id = {}

    raw_label_counts = Counter()

    skipped_nei = 0
    duplicate_rows = 0

    for row in dataset:

        claim_id = int(row["id"])

        claim = str(row["claim"]).strip()

        fever_label = str(row["label"]).strip()

        raw_label_counts[fever_label] += 1

        if fever_label not in LABEL_MAP:

            if fever_label == "NOT ENOUGH INFO":

                skipped_nei += 1
                continue

            raise ValueError(
                f"{split_name}: "
                f"unknown FEVER label "
                f"{fever_label!r}"
            )

        if not claim:

            raise ValueError(
                f"{split_name}: "
                f"empty claim "
                f"for claim_id={claim_id}"
            )

        existing = claims_by_id.get(
            claim_id
        )

        if existing is None:
            claims_by_id[claim_id] = {
                "claim_id": claim_id,
                "claim": claim,
                "fever_label": fever_label,
                "label": LABEL_MAP[
                    fever_label
                ],
                "split": split_name,
            }

            continue

        if existing["claim"] != claim:
            raise ValueError(
                f"{split_name}: "
                f"claim_id={claim_id} "
                f"has inconsistent claim text"
            )

        if (
            existing["fever_label"]
            != fever_label
        ):
            raise ValueError(
                f"{split_name}: "
                f"claim_id={claim_id} "
                f"has inconsistent labels: "
                f"{existing['fever_label']} "
                f"vs "
                f"{fever_label}"
            )

        duplicate_rows += 1

    audit_records = list(
        claims_by_id.values()
    )

    fire_records = [
        {
            "claim": record["claim"],
            "label": record["label"],
        }
        for record in audit_records
    ]

    claim_text_counts = Counter(
        record["claim"]
        for record in audit_records
    )

    duplicate_claim_text_groups = sum(
        1
        for count
        in claim_text_counts.values()
        if count > 1
    )

    binary_label_counts = Counter(
        record["label"]
        for record in fire_records
    )
    stats = {
        "split": split_name,
        "raw_rows":
            len(dataset),
        "raw_label_counts":
            dict(raw_label_counts),
        "removed_nei_rows":
            skipped_nei,
        "flattened_duplicate_rows":
            duplicate_rows,
        "binary_unique_claims":
            len(fire_records),
        "binary_label_counts":
            dict(binary_label_counts),
        "duplicate_claim_text_groups":
            duplicate_claim_text_groups,
    }

    return (
        fire_records,
        audit_records,
        stats,
    )

def write_jsonl(
    records,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

def validate_fire_records(
    records,
    split_name: str,
):
    for index, record in enumerate(
        records
    ):
        if set(record) != {
            "claim",
            "label",
        }:
            raise ValueError(
                f"{split_name}[{index}] "
                f"has invalid fields"
            )

        if (
            not isinstance(
                record["claim"],
                str,
            )
            or not record[
                "claim"
            ].strip()
        ):
            raise ValueError(
                f"{split_name}[{index}] "
                f"has invalid claim"
            )

        if record["label"] not in {
            "True",
            "False",
        }:
            raise ValueError(
                f"{split_name}[{index}] "
                f"has invalid label: "
                f"{record['label']}"
            )

def check_split_leakage(audit_by_split,):
    ids = {
        split_name: {
            record["claim_id"] for record in records
        }
        for split_name, records in audit_by_split.items()
    }

    split_names = list(
        ids.keys()
    )

    for i in range(len(split_names)):

        for j in range(i + 1, len(split_names),):
            left = split_names[i]
            right = split_names[j]

            overlap = (
                ids[left] & ids[right]
            )

            if overlap:
                raise ValueError(
                    f"Claim ID leakage "
                    f"between "
                    f"{left} and {right}: "
                    f"{len(overlap)} claims"
                )
    print(
        "\nNo claim_id leakage "
        "between splits."
    )

def print_statistics(stats):
    print(
        f"\nSplit: "
        f"{stats['split']}"
    )
    print(
        "-" * 50
    )
    print(
        "Raw rows:",
        stats["raw_rows"],
    )

    print(
        "Raw label counts:",
        stats[
            "raw_label_counts"
        ],
    )

    print(
        "Removed NEI rows:",
        stats[
            "removed_nei_rows"
        ],
    )

    print(
        "Flattened duplicate rows:",
        stats[
            "flattened_duplicate_rows"
        ],
    )

    print(
        "Binary unique claims:",
        stats[
            "binary_unique_claims"
        ],
    )

    true_count = (
        stats[
            "binary_label_counts"
        ].get(
            "True",
            0,
        )
    )

    false_count = (
        stats[
            "binary_label_counts"
        ].get(
            "False",
            0,
        )
    )

    print(
        "True:",
        true_count,
    )

    print(
        "False:",
        false_count,
    )

    print(
        "Duplicate claim text groups:",
        stats[
            "duplicate_claim_text_groups"
        ],
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/fever"
        ),
    )
    args = parser.parse_args()
    output_root = Path(args.output_dir)

    fire_dir = (output_root/"fire")
    audit_dir = (output_root/"audit")

    dataset = load_fever_dataset()

    audit_by_split = {}

    all_stats = {}

    for split_name in FEVER_FILES:

        print(
            f"\nProcessing "
            f"{split_name}..."
        )
        (
            fire_records,
            audit_records,
            stats,
        ) = prepare_split(
            dataset[split_name],
            split_name,
        )

        validate_fire_records(
            fire_records,
            split_name,
        )

        write_jsonl(
            fire_records,
            fire_dir/f"{split_name}.jsonl",
        )

        write_jsonl(
            audit_records,
            audit_dir/f"{split_name}.jsonl",
        )

        audit_by_split[split_name] = audit_records

        all_stats[split_name] = stats

        print_statistics(stats)

    check_split_leakage(audit_by_split)
    manifest = {
        "dataset":FEVER_REPO,
        "revision":FEVER_REVISION,
        "source_files":FEVER_FILES,
        "label_mapping":LABEL_MAP,
        "excluded_labels": [
            "NOT ENOUGH INFO"
        ],
        "splits":
            all_stats,
    }

    manifest_path = (
        output_root/"manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\nProcessing finished.")

    print(
        f"FIRE data: "
        f"{fire_dir}"
    )

    print(
        f"Audit data: "
        f"{audit_dir}"
    )

    print(
        f"Manifest: "
        f"{manifest_path}"
    )

if __name__ == "__main__":
    main()