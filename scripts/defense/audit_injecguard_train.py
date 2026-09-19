from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import requests

TRAIN_URL = (
    "https://raw.githubusercontent.com/"
    "leolee99/PIGuard/main/datasets/train.json"
)

def download_file(
    url: str,
    destination: Path,
) -> None:
    if destination.exists():
        print(f"Already exists: {destination}")
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Downloading:\n  {url}")

    response = requests.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    destination.write_bytes(
        response.content
    )

    print(f"Saved: {destination}")

def normalize_text(text: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = " ".join(
        text.strip().split()
    )

    return text.casefold()

def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Expected train.json "
            "to contain a JSON list."
        )

    return data

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

def audit(records: list[dict]):

    label_counter = Counter()
    source_counter = Counter()
    schema_counter = Counter()

    empty_prompt = []
    invalid_label = []
    missing_fields = []

    exact_groups = defaultdict(list)
    normalized_groups = defaultdict(list)

    for index, record in enumerate(records):

        if not isinstance(record, dict):
            missing_fields.append(
                {
                    "index": index,
                    "reason": "record_not_object",
                }
            )

            continue

        schema_counter[
            tuple(sorted(record.keys()))
        ] += 1

        required = {
            "prompt",
            "label",
        }

        missing = (
            required
            -
            set(record.keys())
        )

        if missing:

            missing_fields.append(
                {
                    "index": index,
                    "missing": sorted(missing),
                }
            )

            continue

        prompt = record["prompt"]
        label = record["label"]

        source = record.get(
            "source",
            "<missing>",
        )

        label_counter[
            str(label)
        ] += 1

        source_counter[
            str(source)
        ] += 1

        if (
            not isinstance(prompt, str)
            or
            not prompt.strip()
        ):

            empty_prompt.append(
                {
                    "index": index,
                    "label": label,
                    "source": source,
                }
            )

            continue

        if label not in {
            0,
            1,
            False,
            True,
        }:

            invalid_label.append(
                {
                    "index": index,
                    "label": label,
                    "source": source,
                }
            )

        exact_groups[
            prompt
        ].append(
            {
                "index": index,
                "label": int(label),
                "source": source,
            }
        )

        normalized_groups[
            normalize_text(prompt)
        ].append(
            {
                "index": index,
                "label": int(label),
                "source": source,
                "prompt": prompt,
            }
        )

    exact_duplicates = []
    exact_conflicts = []

    for text, items in exact_groups.items():

        if len(items) <= 1:
            continue

        labels = {
            item["label"]
            for item in items
        }

        record = {
            "text_sha256": sha256_text(text),
            "count": len(items),
            "labels": sorted(labels),
            "sources": sorted({
                str(item["source"])
                for item in items
            }),
            "indices": [
                item["index"]
                for item in items
            ],
        }

        exact_duplicates.append(
            record
        )

        if len(labels) > 1:

            exact_conflicts.append(
                record
            )

    normalized_duplicates = []
    normalized_conflicts = []

    for normalized, items in normalized_groups.items():

        if len(items) <= 1:
            continue

        labels = {
            item["label"]
            for item in items
        }

        record = {
            "normalized_sha256":
                sha256_text(normalized),

            "count":
                len(items),

            "labels":
                sorted(labels),

            "sources":
                sorted({
                    str(item["source"])
                    for item in items
                }),

            "indices":
                [
                    item["index"]
                    for item in items
                ],

            "example":
                items[0]["prompt"][:500],
        }

        normalized_duplicates.append(
            record
        )

        if len(labels) > 1:

            normalized_conflicts.append(
                record
            )

    report = {
        "total_records":
            len(records),

        "label_distribution":
            dict(label_counter),

        "source_distribution":
            dict(source_counter.most_common()),

        "schema_distribution": [
            {
                "fields": list(fields),
                "count": count,
            }
            for fields, count
            in schema_counter.most_common()
        ],

        "empty_prompt_count":
            len(empty_prompt),

        "invalid_label_count":
            len(invalid_label),

        "missing_field_count":
            len(missing_fields),

        "exact_duplicate_groups":
            len(exact_duplicates),

        "exact_label_conflict_groups":
            len(exact_conflicts),

        "normalized_duplicate_groups":
            len(normalized_duplicates),

        "normalized_label_conflict_groups":
            len(normalized_conflicts),
    }

    diagnostics = {
        "empty_prompts":
            empty_prompt,

        "invalid_labels":
            invalid_label,

        "missing_fields":
            missing_fields,

        "exact_duplicates":
            exact_duplicates,

        "exact_conflicts":
            exact_conflicts,

        "normalized_duplicates":
            normalized_duplicates,

        "normalized_conflicts":
            normalized_conflicts,
    }

    return (
        report,
        diagnostics,
    )


def print_report(report: dict) -> None:

    print("\n================================")
    print("InjecGuard/PIGuard Basic Audit")
    print("================================")

    print(
        "\nTotal records:",
        report["total_records"],
    )

    print(
        "\nLabel distribution:"
    )

    for label, count in (
        report[
            "label_distribution"
        ].items()
    ):

        meaning = (
            "benign"
            if label in {"0", "False"}
            else "injection"
            if label in {"1", "True"}
            else "UNKNOWN"
        )

        print(
            f"  {label}: "
            f"{count} ({meaning})"
        )

    print(
        "\nSources:"
    )

    for source, count in (
        report[
            "source_distribution"
        ].items()
    ):

        print(
            f"  {source}: {count}"
        )

    print(
        "\nSchema variants:",
        len(
            report[
                "schema_distribution"
            ]
        ),
    )

    print(
        "Empty prompts:",
        report[
            "empty_prompt_count"
        ],
    )

    print(
        "Invalid labels:",
        report[
            "invalid_label_count"
        ],
    )

    print(
        "Missing required fields:",
        report[
            "missing_field_count"
        ],
    )

    print(
        "\nExact duplicate groups:",
        report[
            "exact_duplicate_groups"
        ],
    )

    print(
        "Exact label conflicts:",
        report[
            "exact_label_conflict_groups"
        ],
    )

    print(
        "Normalized duplicate groups:",
        report[
            "normalized_duplicate_groups"
        ],
    )

    print(
        "Normalized label conflicts:",
        report[
            "normalized_label_conflict_groups"
        ],
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
        "--report",
        default=(
            "data/processed/defense/"
            "audit/injecguard/"
            "basic_audit.json"
        ),
    )

    parser.add_argument(
        "--diagnostics-dir",
        default=(
            "data/processed/defense/"
            "audit/injecguard"
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    report_path = Path(
        args.report
    )

    diagnostics_dir = Path(
        args.diagnostics_dir
    )

    download_file(
        TRAIN_URL,
        input_path,
    )

    records = load_json(
        input_path
    )

    (
        report,
        diagnostics,
    ) = audit(
        records
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for name, items in diagnostics.items():

        write_jsonl(
            items,
            diagnostics_dir
            /
            f"{name}.jsonl",
        )

    print_report(
        report
    )

    print(
        "\nReport:",
        report_path,
    )

if __name__ == "__main__":
    main()