from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path


HARD_CONTAMINATION_REASONS = {
    "notinject_full_text",
    "bipia_full_text",
    "bipia_test_payload",
}


def normalize_text(text: str) -> str:

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    return " ".join(
        text.strip().split()
    ).casefold()


def sha256_text(text: str) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        return []

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:

                records.append(
                    json.loads(line)
                )

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


def load_source_registry(
    path: Path,
) -> dict[str, dict]:

    data = load_json(path)

    if not isinstance(data, list):

        raise ValueError(
            "Source registry must "
            "contain a JSON list."
        )

    return {
        item["source"]: item
        for item in data
    }


def load_hard_contamination(
    path: Path,
) -> dict[int, list[str]]:

    records = load_jsonl(
        path
    )

    result = {}

    for record in records:

        reasons = set(
            record.get(
                "reasons",
                [],
            )
        )

        hard_reasons = sorted(
            reasons
            &
            HARD_CONTAMINATION_REASONS
        )

        if hard_reasons:

            result[
                int(record["index"])
            ] = hard_reasons

    return result


def find_conflict_groups(
    records: list[dict],
) -> set[str]:

    groups = defaultdict(
        set
    )

    for record in records:

        prompt = record.get(
            "prompt"
        )

        label = record.get(
            "label"
        )

        if (
            not isinstance(prompt, str)
            or
            not prompt.strip()
            or
            label not in {0, 1}
        ):

            continue

        key = normalize_text(
            prompt
        )

        groups[
            key
        ].add(
            int(label)
        )

    return {
        key
        for key, labels
        in groups.items()
        if len(labels) > 1
    }


def convert_record(
    index: int,
    record: dict,
    registry_entry: dict,
) -> dict:

    prompt = record["prompt"]

    normalized = normalize_text(
        prompt
    )

    label = (
        "benign"
        if int(record["label"]) == 0
        else "injection"
    )

    source = str(
        record.get(
            "source",
            "<missing>",
        )
    )

    return {
        "sample_id":
            f"injecguard_{index:06d}",

        "text":
            prompt,

        "label":
            label,

        "source":
            "injecguard",

        "original_source":
            source,

        "source_record_id":
            str(index),

        "source_semantic_type":
            registry_entry.get(
                "semantic_type"
            ),

        "claim_id":
            None,

        "evidence_refs":
            None,

        "granularity":
            "prompt",

        "base_text_id":
            None,

        "base_text_sha256":
            None,

        "text_sha256":
            sha256_text(
                prompt
            ),

        "normalized_text_sha256":
            sha256_text(
                normalized
            ),

        "attack_mode":
            None,

        "attack_family":
            None,

        "attack_objective":
            None,

        "template_id":
            None,

        "template_split":
            None,

        "insertion_position":
            None,

        "representation":
            None,

        "encoding":
            None,

        "language":
            None,

        "synthetic":
            (
                source
                == "LLM Augmented set"
            ),

        "split":
            "train",
    }


def clean_records(
    records: list[dict],
    registry: dict[str, dict],
    hard_contamination: dict[int, list[str]],
):

    conflict_keys = find_conflict_groups(
        records
    )

    clean_candidates = []
    quarantine = []

    rejected_source = []
    rejected_empty = []
    rejected_conflict = []
    rejected_contamination = []
    rejected_invalid = []

    for index, record in enumerate(records):

        if not isinstance(record, dict):

            rejected_invalid.append(
                {
                    "index": index,
                    "reason":
                        "record_not_object",
                }
            )

            continue

        prompt = record.get(
            "prompt"
        )

        label = record.get(
            "label"
        )

        source = str(
            record.get(
                "source",
                "<missing>",
            )
        )

        if (
            not isinstance(prompt, str)
            or
            not prompt.strip()
        ):

            rejected_empty.append(
                {
                    "index":
                        index,

                    "source":
                        source,

                    "label":
                        label,

                    "reason":
                        "empty_prompt",
                }
            )

            continue

        if label not in {0, 1}:

            rejected_invalid.append(
                {
                    "index":
                        index,

                    "source":
                        source,

                    "label":
                        label,

                    "reason":
                        "invalid_label",
                }
            )

            continue

        normalized = normalize_text(
            prompt
        )

        if normalized in conflict_keys:

            rejected_conflict.append(
                {
                    "index":
                        index,

                    "source":
                        source,

                    "label":
                        label,

                    "prompt":
                        prompt,

                    "normalized_text_sha256":
                        sha256_text(
                            normalized
                        ),

                    "reason":
                        "normalized_label_conflict",
                }
            )

            continue

        if index in hard_contamination:

            rejected_contamination.append(
                {
                    "index":
                        index,

                    "source":
                        source,

                    "label":
                        label,

                    "prompt":
                        prompt,

                    "reasons":
                        hard_contamination[
                            index
                        ],
                }
            )

            continue

        registry_entry = registry.get(
            source
        )

        if registry_entry is None:

            quarantine.append(
                {
                    "index":
                        index,

                    "source":
                        source,

                    "label":
                        label,

                    "prompt":
                        prompt,

                    "reason":
                        "source_missing_from_registry",
                }
            )

            continue

        decision = registry_entry.get(
            "decision",
            "review",
        )

        if decision == "remove":

            rejected_source.append(
                {
                    "index":
                        index,

                    "source":
                        source,

                    "label":
                        label,

                    "prompt":
                        prompt,

                    "reason":
                        "source_policy_remove",
                }
            )

            continue

        if decision == "review":

            quarantine.append(
                {
                    "index":
                        index,

                    "source":
                        source,

                    "label":
                        label,

                    "prompt":
                        prompt,

                    "semantic_type":
                        registry_entry.get(
                            "semantic_type"
                        ),

                    "contamination_risk":
                        registry_entry.get(
                            "contamination_risk"
                        ),

                    "reason":
                        "source_policy_review",
                }
            )

            continue

        if decision != "candidate_keep":

            quarantine.append(
                {
                    "index":
                        index,

                    "source":
                        source,

                    "label":
                        label,

                    "prompt":
                        prompt,

                    "reason":
                        f"unknown_decision:{decision}",
                }
            )

            continue

        clean_candidates.append(
            (
                index,
                record,
                registry_entry,
            )
        )

    clean = []
    duplicates = []

    seen = {}

    for (
        index,
        record,
        registry_entry,
    ) in clean_candidates:

        prompt = record[
            "prompt"
        ]

        label = int(
            record["label"]
        )

        normalized = normalize_text(
            prompt
        )

        key = (
            normalized,
            label,
        )

        if key in seen:

            duplicates.append(
                {
                    "index":
                        index,

                    "source":
                        record.get(
                            "source"
                        ),

                    "label":
                        label,

                    "prompt":
                        prompt,

                    "normalized_text_sha256":
                        sha256_text(
                            normalized
                        ),

                    "kept_index":
                        seen[key],

                    "reason":
                        "normalized_duplicate_same_label",
                }
            )

            continue

        seen[
            key
        ] = index

        clean.append(
            convert_record(
                index,
                record,
                registry_entry,
            )
        )

    return {
        "clean":
            clean,

        "quarantine":
            quarantine,

        "rejected_source":
            rejected_source,

        "rejected_empty":
            rejected_empty,

        "rejected_conflict":
            rejected_conflict,

        "rejected_contamination":
            rejected_contamination,

        "rejected_invalid":
            rejected_invalid,

        "duplicates":
            duplicates,
    }


def build_report(
    original_count: int,
    result: dict,
) -> dict:

    clean = result[
        "clean"
    ]

    benign = sum(
        record["label"] == "benign"
        for record in clean
    )

    injection = sum(
        record["label"] == "injection"
        for record in clean
    )

    source_distribution = defaultdict(
        int
    )

    for record in clean:

        source_distribution[
            record[
                "original_source"
            ]
        ] += 1

    return {
        "input_records":
            original_count,

        "clean_records":
            len(clean),

        "clean_benign":
            benign,

        "clean_injection":
            injection,

        "quarantined_review":
            len(
                result[
                    "quarantine"
                ]
            ),

        "removed_source_policy":
            len(
                result[
                    "rejected_source"
                ]
            ),

        "removed_empty":
            len(
                result[
                    "rejected_empty"
                ]
            ),

        "removed_label_conflict":
            len(
                result[
                    "rejected_conflict"
                ]
            ),

        "removed_hard_contamination":
            len(
                result[
                    "rejected_contamination"
                ]
            ),

        "removed_invalid":
            len(
                result[
                    "rejected_invalid"
                ]
            ),

        "removed_duplicates":
            len(
                result[
                    "duplicates"
                ]
            ),

        "clean_source_distribution":
            dict(
                sorted(
                    source_distribution.items()
                )
            ),
    }


def print_report(
    report: dict,
) -> None:

    print(
        "\n================================"
    )

    print(
        "InjecGuard Decontamination"
    )

    print(
        "================================"
    )

    print(
        "\nInput records:",
        report[
            "input_records"
        ],
    )

    print(
        "Clean records:",
        report[
            "clean_records"
        ],
    )

    print(
        "  benign:",
        report[
            "clean_benign"
        ],
    )

    print(
        "  injection:",
        report[
            "clean_injection"
        ],
    )

    print(
        "\nQuarantined review:",
        report[
            "quarantined_review"
        ],
    )

    print(
        "Removed source policy:",
        report[
            "removed_source_policy"
        ],
    )

    print(
        "Removed empty:",
        report[
            "removed_empty"
        ],
    )

    print(
        "Removed conflicts:",
        report[
            "removed_label_conflict"
        ],
    )

    print(
        "Removed hard contamination:",
        report[
            "removed_hard_contamination"
        ],
    )

    print(
        "Removed invalid:",
        report[
            "removed_invalid"
        ],
    )

    print(
        "Removed duplicates:",
        report[
            "removed_duplicates"
        ],
    )

    print(
        "\nClean sources:"
    )

    for source, count in (
        report[
            "clean_source_distribution"
        ].items()
    ):

        print(
            f"  {source}: {count}"
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/raw/injecguard/"
            "train.json"
        ),
    )

    parser.add_argument(
        "--registry",
        default=(
            "data/processed/defense/"
            "audit/injecguard/"
            "source_registry_draft.json"
        ),
    )

    parser.add_argument(
        "--contamination",
        default=(
            "data/processed/defense/"
            "audit/injecguard/"
            "contaminated_samples.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/defense/"
            "sources"
        ),
    )

    parser.add_argument(
        "--audit-dir",
        default=(
            "data/processed/defense/"
            "audit/injecguard/"
            "decontamination"
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    registry_path = Path(
        args.registry
    )

    contamination_path = Path(
        args.contamination
    )

    output_dir = Path(
        args.output_dir
    )

    audit_dir = Path(
        args.audit_dir
    )

    records = load_json(
        input_path
    )

    if not isinstance(
        records,
        list,
    ):

        raise ValueError(
            "train.json must contain a list."
        )

    registry = load_source_registry(
        registry_path
    )

    hard_contamination = (
        load_hard_contamination(
            contamination_path
        )
    )

    result = clean_records(
        records,
        registry,
        hard_contamination,
    )

    report = build_report(
        len(records),
        result,
    )

    write_jsonl(
        result["clean"],
        output_dir
        /
        "injecguard_clean.jsonl",
    )

    write_jsonl(
        result["quarantine"],
        output_dir
        /
        "injecguard_quarantine.jsonl",
    )

    for name in (
        "rejected_source",
        "rejected_empty",
        "rejected_conflict",
        "rejected_contamination",
        "rejected_invalid",
        "duplicates",
    ):

        write_jsonl(
            result[name],
            audit_dir
            /
            f"{name}.jsonl",
        )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        audit_dir
        /
        "decontamination_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_report(
        report
    )

    print(
        "\nClean dataset:",
        output_dir
        /
        "injecguard_clean.jsonl",
    )

    print(
        "Quarantine:",
        output_dir
        /
        "injecguard_quarantine.jsonl",
    )

    print(
        "Report:",
        report_path,
    )

if __name__ == "__main__":
    main()