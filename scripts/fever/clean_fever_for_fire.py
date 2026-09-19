from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

SPLITS = (
    "train",
    "paper_dev",
    "paper_test",
)

def load_jsonl(path: Path) -> list[dict]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue
            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON: "
                    f"{path}:{line_number}"
                ) from error

    return records

def write_jsonl(
    records: list[dict],
    path: Path,
) -> None:
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

def normalize_claim(text: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        text,
    )
    text = text.strip()
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.casefold()

def clean_split(
    records: list[dict],
    split_name: str,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    dict,
]:
    groups = defaultdict(list)

    for record in records:
        key = normalize_claim(
            record["claim"]
        )
        groups[key].append(record)

    clean_records = []
    conflict_records = []
    duplicate_records = []
    conflict_groups = 0
    duplicate_groups = 0

    for key, group in groups.items():

        labels = {
            record["label"]
            for record in group
        }

        if len(labels) > 1:
            conflict_groups += 1
            for record in group:
                rejected = dict(record)
                rejected[
                    "rejection_reason"
                ] = "label_conflict"
                conflict_records.append(
                    rejected
                )

            continue

        if len(group) > 1:

            duplicate_groups += 1

        group = sorted(
            group,
            key=lambda item: item["claim_id"],
        )

        keep = group[0]

        clean_records.append(
            keep
        )

        for record in group[1:]:

            rejected = dict(record)

            rejected[
                "rejection_reason"
            ] = "duplicate_same_label"

            duplicate_records.append(
                rejected
            )

    clean_records.sort(
        key=lambda item: item["claim_id"]
    )

    stats = {
        "split":
            split_name,
        "input_claims":
            len(records),
        "output_claims":
            len(clean_records),
        "duplicate_groups":
            duplicate_groups,
        "removed_duplicate_records":
            len(duplicate_records),
        "conflict_groups":
            conflict_groups,
        "removed_conflict_records":
            len(conflict_records),
    }

    return (
        clean_records,
        duplicate_records,
        conflict_records,
        stats,
    )

def remove_train_leakage(
    cleaned: dict[str, list[dict]],
) -> tuple[
    dict[str, list[dict]],
    list[dict],
]:
    train = cleaned["train"]
    dev = cleaned["paper_dev"]
    test = cleaned["paper_test"]

    dev_keys = {
        normalize_claim(
            record["claim"]
        )
        for record in dev
    }

    test_keys = {
        normalize_claim(
            record["claim"]
        )
        for record in test
    }

    dev_test_overlap = (dev_keys & test_keys)
    if dev_test_overlap:
        raise ValueError(
            "paper_dev and paper_test "
            f"contain "
            f"{len(dev_test_overlap)} "
            "overlapping claims."
        )

    protected_keys = (dev_keys|test_keys)

    clean_train = []

    leaked_train_records = []

    for record in train:
        key = normalize_claim(
            record["claim"]
        )
        if key in protected_keys:
            rejected = dict(record)
            rejected[
                "rejection_reason"
            ] = "cross_split_leakage"
            leaked_train_records.append(
                rejected
            )
            continue
        clean_train.append(
            record
        )
    cleaned["train"] = clean_train

    return (
        cleaned,
        leaked_train_records,
    )

def make_fire_records(
    audit_records: list[dict],
) -> list[dict]:

    return [
        {
            "claim":
                record["claim"],
            "label":
                record["label"],
        }
        for record in audit_records
    ]

def validate_clean_dataset(
    datasets: dict[str, list[dict]],
) -> None:
    normalized_sets = {}
    for split_name, records in datasets.items():
        seen = set()
        for record in records:
            key = normalize_claim(
                record["claim"]
            )
            if key in seen:

                raise ValueError(
                    f"Duplicate remained "
                    f"in {split_name}: "
                    f"{record['claim']}"
                )

            seen.add(key)

        normalized_sets[
            split_name
        ] = seen

    for i, left in enumerate(SPLITS):
        for right in SPLITS[
            i + 1:
        ]:
            overlap = (
                normalized_sets[left]
                &
                normalized_sets[right]
            )
            if overlap:
                raise ValueError(
                    f"Cross-split leakage "
                    f"remains between "
                    f"{left} and {right}: "
                    f"{len(overlap)}"
                )

def print_stats(stats: dict) -> None:
    print(
        f"\nSplit: "
        f"{stats['split']}"
    )
    print(
        "-" * 50
    )
    print(
        "Input claims:",
        stats["input_claims"],
    )
    print(
        "Output before leakage removal:",
        stats["output_claims"],
    )
    print(
        "Duplicate groups:",
        stats["duplicate_groups"],
    )
    print(
        "Removed duplicate records:",
        stats[
            "removed_duplicate_records"
        ],
    )
    print(
        "Conflict groups:",
        stats["conflict_groups"],
    )
    print(
        "Removed conflict records:",
        stats[
            "removed_conflict_records"
        ],
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=(
            "data/processed/"
            "fever/audit"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/"
            "fever/clean"
        ),
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    cleaned = {}
    all_stats = {}
    all_duplicates = {}
    all_conflicts = {}

    for split_name in SPLITS:
        print(
            f"\nCleaning "
            f"{split_name}..."
        )

        records = load_jsonl(input_dir/f"{split_name}.jsonl")

        (
            clean_records,
            duplicate_records,
            conflict_records,
            stats,

        ) = clean_split(
            records,
            split_name,
        )

        cleaned[split_name] = clean_records
        all_stats[split_name] = stats
        all_duplicates[split_name] = duplicate_records
        all_conflicts[split_name] = conflict_records
        print_stats(stats)

    (cleaned,leakage_records) = remove_train_leakage(cleaned)

    print(
        "\nRemoved train cross-split "
        f"leakage records: "
        f"{len(leakage_records)}"
    )

    validate_clean_dataset(
        cleaned
    )

    for split_name in SPLITS:
        audit_records = (
            cleaned[split_name]
        )
        fire_records = (
            make_fire_records(
                audit_records
            )
        )

        write_jsonl(audit_records,output_dir/"audit"/f"{split_name}.jsonl")
        write_jsonl(fire_records,output_dir/"fire"/f"{split_name}.jsonl")
        write_jsonl(all_duplicates[split_name],output_dir/"rejected"/f"{split_name}_duplicates.jsonl")
        write_jsonl(all_conflicts[split_name],output_dir/"rejected"/f"{split_name}_conflicts.jsonl")

    write_jsonl(leakage_records, output_dir/"rejected"/"train_cross_split_leakage.jsonl")

    final_stats = {}

    for split_name in SPLITS:

        records = cleaned[
            split_name
        ]

        true_count = sum(
            1
            for record in records
            if record["label"] == "True"
        )

        false_count = sum(
            1
            for record in records
            if record["label"] == "False"
        )

        final_stats[
            split_name
        ] = {
            "claims":len(records),
            "true":true_count,
            "false":false_count,
        }

    report = {
        "cleaning_policy": {
            "normalization":
                "NFKC + strip + "
                "collapse whitespace + casefold",
            "same_label_duplicates":
                "keep lowest claim_id",
            "label_conflicts":
                "remove entire group",
            "cross_split_overlap":
                "protect paper_dev/"
                "paper_test; remove from train",
        },
        "processing":
            all_stats,
        "removed_cross_split":
            len(leakage_records),
        "final":
            final_stats,
    }

    report_path = (output_dir/"cleaning_report.json")

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nFinal clean dataset:")
    print("-" * 50)

    for split_name in SPLITS:
        stats = final_stats[
            split_name
        ]
        print(
            f"{split_name}: "
            f"{stats['claims']} claims | "
            f"True={stats['true']} | "
            f"False={stats['false']}"
        )

    print("\nCleaning completed successfully.")
    print("Output:",output_dir)
    print("Report:",report_path)
if __name__ == "__main__":
    main()