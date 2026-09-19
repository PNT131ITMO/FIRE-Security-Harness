from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
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

                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}"
                ) from error
            records.append(record)

    return records

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

def validate_fire_schema(
    records: list[dict],
    split_name: str,
) -> None:
    for index, record in enumerate(records):
        if set(record.keys()) != {
            "claim",
            "label",
        }:
            raise ValueError(
                f"{split_name}: "
                f"invalid FIRE fields "
                f"at index {index}: "
                f"{set(record.keys())}"
            )
        claim = record["claim"]
        label = record["label"]
        if (
            not isinstance(claim, str)
            or not claim.strip()
        ):
            raise ValueError(
                f"{split_name}: "
                f"invalid claim "
                f"at index {index}"
            )
        if label not in {
            "True",
            "False",
        }:
            raise ValueError(
                f"{split_name}: "
                f"invalid label "
                f"at index {index}: "
                f"{label!r}"
            )

def validate_audit_schema(
    records: list[dict],
    split_name: str,
) -> None:
    required_fields = {
        "claim_id",
        "claim",
        "fever_label",
        "label",
        "split",
    }
    for index, record in enumerate(records):
        missing = (
            required_fields
            -
            set(record.keys())
        )
        if missing:
            raise ValueError(
                f"{split_name}: "
                f"audit record {index} "
                f"is missing fields: "
                f"{missing}"
            )
        if record["label"] not in {
            "True",
            "False",
        }:
            raise ValueError(
                f"{split_name}: "
                f"invalid audit label "
                f"at index {index}"
            )
        if record["fever_label"] not in {
            "SUPPORTS",
            "REFUTES",
        }:
            raise ValueError(
                f"{split_name}: "
                f"invalid FEVER label "
                f"at index {index}"
            )
        if record["split"] != split_name:
            raise ValueError(
                f"{split_name}: "
                f"wrong split field "
                f"at index {index}: "
                f"{record['split']}"
            )

def check_fire_audit_consistency(
    fire_records: list[dict],
    audit_records: list[dict],
    split_name: str,
) -> None:
    if len(fire_records) != len(audit_records):
        raise ValueError(
            f"{split_name}: "
            f"FIRE/audit length mismatch: "
            f"{len(fire_records)} vs "
            f"{len(audit_records)}"
        )
    for index, (
        fire_record,
        audit_record,
    ) in enumerate(
        zip(
            fire_records,
            audit_records,
        )
    ):
        if (
            fire_record["claim"]
            != audit_record["claim"]
        ):
            raise ValueError(
                f"{split_name}: "
                f"claim mismatch "
                f"at index {index}"
            )
        if (
            fire_record["label"]
            != audit_record["label"]
        ):
            raise ValueError(
                f"{split_name}: "
                f"label mismatch "
                f"at index {index}"
            )

def analyze_duplicates(
    records: list[dict],
) -> dict:
    by_exact_text = defaultdict(list)
    by_normalized_text = defaultdict(list)
    for record in records:
        claim = record["claim"]
        by_exact_text[claim].append(record)
        normalized = normalize_claim(claim)
        by_normalized_text[
            normalized
        ].append(record)

    exact_duplicates = {
        claim: group
        for claim, group
        in by_exact_text.items()
        if len(group) > 1
    }

    normalized_duplicates = {
        claim: group
        for claim, group
        in by_normalized_text.items()
        if len(group) > 1
    }

    exact_conflicts = {}
    for claim, group in exact_duplicates.items():
        labels = {
            item["label"]
            for item in group
        }
        if len(labels) > 1:
            exact_conflicts[claim] = group
    normalized_conflicts = {}
    for claim, group in normalized_duplicates.items():
        labels = {
            item["label"]
            for item in group
        }
        if len(labels) > 1:
            normalized_conflicts[
                claim
            ] = group

    return {
        "exact_duplicate_groups":
            len(exact_duplicates),
        "normalized_duplicate_groups":
            len(normalized_duplicates),
        "exact_label_conflicts":
            len(exact_conflicts),
        "normalized_label_conflicts":
            len(normalized_conflicts),
        "exact_duplicate_examples":
            make_examples(
                exact_duplicates
            ),
        "exact_conflict_examples":
            make_examples(
                exact_conflicts
            ),
        "normalized_conflict_examples":
            make_examples(
                normalized_conflicts
            ),
    }

def make_examples(
    groups: dict,
    limit: int = 10,
) -> list[dict]:
    examples = []
    for claim, records in list(
        groups.items()
    )[:limit]:
        examples.append(
            {
                "claim": claim,
                "records": [
                    {
                        "claim_id":
                            record["claim_id"],
                        "label":
                            record["label"],
                        "fever_label":
                            record[
                                "fever_label"
                            ],
                    }
                    for record in records
                ],
            }
        )

    return examples

def analyze_cross_split_overlap(
    audit_by_split: dict[
        str,
        list[dict],
    ],
) -> dict:
    exact_maps = {}
    normalized_maps = {}

    for split_name, records \
            in audit_by_split.items():
        exact_maps[split_name] = {
            record["claim"]:
                record
            for record in records
        }

        normalized_maps[split_name] = {
            normalize_claim(
                record["claim"]
            ):
                record
            for record in records
        }

    results = {}

    split_names = list(
        audit_by_split.keys()
    )

    for i in range(
        len(split_names)
    ):
        for j in range(
            i + 1,
            len(split_names),
        ):
            left = split_names[i]
            right = split_names[j]

            exact_overlap = (
                set(
                    exact_maps[left]
                )
                &
                set(
                    exact_maps[right]
                )
            )

            normalized_overlap = (
                set(
                    normalized_maps[left]
                )
                &
                set(
                    normalized_maps[right]
                )
            )

            key = (
                f"{left}__{right}"
            )

            examples = []

            for claim in list(
                exact_overlap
            )[:10]:

                left_record = (
                    exact_maps[left][claim]
                )
                right_record = (
                    exact_maps[right][claim]
                )
                examples.append(
                    {
                        "claim": claim,
                        left: {
                            "claim_id":
                                left_record[
                                    "claim_id"
                                ],
                            "label":
                                left_record[
                                    "label"
                                ],
                        },
                        right: {
                            "claim_id":
                                right_record[
                                    "claim_id"
                                ],
                            "label":
                                right_record[
                                    "label"
                                ],
                        },
                    }
                )

            results[key] = {
                "exact_overlap":
                    len(
                        exact_overlap
                    ),
                "normalized_overlap":
                    len(
                        normalized_overlap
                    ),
                "exact_examples":
                    examples,
            }

    return results

def get_label_statistics(
    records: list[dict],
) -> dict:
    counts = Counter(
        record["label"]
        for record in records
    )
    total = len(records)

    return {
        "total": total,
        "true":
            counts.get(
                "True",
                0,
            ),
        "false":
            counts.get(
                "False",
                0,
            ),
        "true_ratio":
            (
                counts.get(
                    "True",
                    0,
                )
                / total
            )

            if total
            else 0,
        "false_ratio":
            (
                counts.get(
                    "False",
                    0,
                )
                / total
            )

            if total
            else 0,
    }

def print_split_report(
    split_name: str,
    label_stats: dict,
    duplicate_stats: dict,
) -> None:
    print(
        f"\nSplit: {split_name}"
    )
    print(
        "-" * 60
    )
    print(
        "Total claims:",
        label_stats["total"],
    )
    print(
        "True:",
        label_stats["true"],
    )
    print(
        "False:",
        label_stats["false"],
    )
    print(
        "True ratio:",
        f"{label_stats['true_ratio']:.4f}",
    )
    print(
        "Exact duplicate groups:",
        duplicate_stats[
            "exact_duplicate_groups"
        ],
    )
    print(
        "Normalized duplicate groups:",
        duplicate_stats[
            "normalized_duplicate_groups"
        ],
    )
    print(
        "Exact label conflicts:",
        duplicate_stats[
            "exact_label_conflicts"
        ],
    )
    print(
        "Normalized label conflicts:",
        duplicate_stats[
            "normalized_label_conflicts"
        ],
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=(
            "data/processed/fever"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "data/processed/fever/"
            "validation_report.json"
        ),
    )
    args = parser.parse_args()

    data_dir = Path(
        args.data_dir
    )
    report_path = Path(
        args.report
    )
    fire_dir = (data_dir/"fire")
    audit_dir = (data_dir/"audit")
    audit_by_split = {}
    report = {
        "splits": {},
        "cross_split_overlap": {},
    }

    for split_name in SPLITS:
        fire_path = (fire_dir/f"{split_name}.jsonl")
        audit_path = (audit_dir/f"{split_name}.jsonl")
        print(
            f"\nValidating "
            f"{split_name}..."
        )
        fire_records = load_jsonl(fire_path)

        audit_records = load_jsonl(audit_path)

        validate_fire_schema(fire_records,split_name,)
        validate_audit_schema(audit_records,split_name,)
        check_fire_audit_consistency(
            fire_records,
            audit_records,
            split_name,
        )

        label_stats = (
            get_label_statistics(
                fire_records
            )
        )

        duplicate_stats = (
            analyze_duplicates(
                audit_records
            )
        )

        report["splits"][
            split_name
        ] = {
            "labels":
                label_stats,

            "duplicates":
                duplicate_stats,
        }

        audit_by_split[
            split_name
        ] = audit_records

        print_split_report(
            split_name,
            label_stats,
            duplicate_stats,
        )

    print(
        "\nChecking "
        "cross-split claim overlap..."
    )

    overlap_report = (
        analyze_cross_split_overlap(
            audit_by_split
        )
    )

    report[
        "cross_split_overlap"
    ] = overlap_report

    for pair, stats \
            in overlap_report.items():

        print(
            f"\n{pair}"
        )

        print(
            "  exact overlap:",
            stats[
                "exact_overlap"
            ],
        )

        print(
            "  normalized overlap:",
            stats[
                "normalized_overlap"
            ],
        )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(

        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),

        encoding="utf-8",
    )

    print(
        "\nValidation completed."
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()