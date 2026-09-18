from __future__ import annotations

import argparse
import json
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
                record = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}"
                ) from error

            records.append(record)

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

def load_bad_sets(
    unresolved_path: Path,
):

    unresolved_records = load_jsonl(
        unresolved_path
    )

    bad_sets = defaultdict(set)

    for item in unresolved_records:

        split = item["split"]

        claim_id = int(
            item["claim_id"]
        )

        evidence_set_index = int(
            item["evidence_set_index"]
        )

        bad_sets[
            (
                split,
                claim_id,
            )
        ].add(
            evidence_set_index
        )

    return (
        bad_sets,
        unresolved_records,
    )

def is_complete_evidence_set(
    evidence_set: list[dict],
) -> bool:

    if not evidence_set:
        return False

    for evidence in evidence_set:

        text = evidence.get(
            "text"
        )

        if not isinstance(
            text,
            str,
        ):
            return False

        if not text.strip():
            return False

        if "page" not in evidence:
            return False

        if "sentence_id" not in evidence:
            return False

    return True

def clean_split(
    split_name: str,
    records: list[dict],
    bad_sets,
):

    clean_records = []

    rejected_claims = []

    removed_sets = []

    affected_but_kept = 0

    removed_set_count = 0

    total_input_sets = 0
    total_output_sets = 0

    seen_claim_ids = set()

    for record in records:

        claim_id = int(
            record["claim_id"]
        )

        if claim_id in seen_claim_ids:

            raise ValueError(
                f"{split_name}: duplicate "
                f"claim_id={claim_id}"
            )

        seen_claim_ids.add(
            claim_id
        )

        evidence_sets = record[
            "evidence_sets"
        ]

        total_input_sets += len(
            evidence_sets
        )

        explicitly_bad = bad_sets.get(
            (
                split_name,
                claim_id,
            ),
            set(),
        )

        valid_sets = []

        removed_indices = []

        for set_index, evidence_set \
                in enumerate(
                    evidence_sets
                ):

            if set_index in explicitly_bad:

                removed_indices.append(
                    set_index
                )

                removed_sets.append(
                    {
                        "split":
                            split_name,

                        "claim_id":
                            claim_id,

                        "claim":
                            record["claim"],

                        "evidence_set_index":
                            set_index,

                        "reason":
                            "contains_unresolved_evidence",
                    }
                )

                removed_set_count += 1

                continue

            if not is_complete_evidence_set(
                evidence_set
            ):

                removed_indices.append(
                    set_index
                )

                removed_sets.append(
                    {
                        "split":
                            split_name,

                        "claim_id":
                            claim_id,

                        "claim":
                            record["claim"],

                        "evidence_set_index":
                            set_index,

                        "reason":
                            "structurally_incomplete_set",
                    }
                )

                removed_set_count += 1

                continue

            valid_sets.append(
                evidence_set
            )

        if not valid_sets:

            rejected_record = {
                "claim_id":
                    claim_id,

                "claim":
                    record["claim"],

                "label":
                    record["label"],

                "fever_label":
                    record["fever_label"],

                "split":
                    split_name,

                "original_evidence_sets":
                    len(
                        evidence_sets
                    ),

                "removed_evidence_set_indices":
                    removed_indices,

                "rejection_reason":
                    "no_complete_evidence_set",
            }

            rejected_claims.append(
                rejected_record
            )

            continue

        clean_record = dict(
            record
        )

        clean_record[
            "evidence_sets"
        ] = valid_sets

        clean_records.append(
            clean_record
        )

        total_output_sets += len(
            valid_sets
        )

        if removed_indices:

            affected_but_kept += 1

    for record in clean_records:

        if not record[
            "evidence_sets"
        ]:

            raise RuntimeError(
                f"{split_name}: claim "
                f"{record['claim_id']} "
                f"has no evidence sets."
            )

        for evidence_set \
                in record[
                    "evidence_sets"
                ]:

            if not is_complete_evidence_set(
                evidence_set
            ):

                raise RuntimeError(
                    f"{split_name}: incomplete "
                    f"evidence survived cleaning "
                    f"for claim_id="
                    f"{record['claim_id']}"
                )

    true_count = sum(
        record["label"] == "True"
        for record in clean_records
    )

    false_count = sum(
        record["label"] == "False"
        for record in clean_records
    )

    stats = {
        "input_claims":
            len(records),

        "output_claims":
            len(clean_records),

        "rejected_claims":
            len(rejected_claims),

        "affected_but_kept":
            affected_but_kept,

        "input_evidence_sets":
            total_input_sets,

        "output_evidence_sets":
            total_output_sets,

        "removed_evidence_sets":
            removed_set_count,

        "true":
            true_count,

        "false":
            false_count,
    }

    return (
        clean_records,
        rejected_claims,
        removed_sets,
        stats,
    )

def validate_cross_split_ids(
    clean_by_split: dict[str, list[dict]],
) -> None:

    ids = {
        split: {
            int(record["claim_id"])
            for record in records
        }
        for split, records
        in clean_by_split.items()
    }

    for i, split_a in enumerate(
        SPLITS
    ):

        for split_b in SPLITS[
            i + 1:
        ]:

            overlap = (
                ids[split_a]
                &
                ids[split_b]
            )

            if overlap:

                raise RuntimeError(
                    f"Cross-split claim_id "
                    f"overlap detected: "
                    f"{split_a} <-> "
                    f"{split_b}: "
                    f"{len(overlap)}"
                )

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resolved-dir",
        default=(
            "data/processed/fever/"
            "gold/resolved"
        ),
    )

    parser.add_argument(
        "--unresolved",
        default=(
            "data/processed/fever/"
            "gold/unresolved_evidence.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/fever/"
            "gold/clean"
        ),
    )

    args = parser.parse_args()

    resolved_dir = Path(
        args.resolved_dir
    )

    unresolved_path = Path(
        args.unresolved
    )

    output_dir = Path(
        args.output_dir
    )

    rejected_dir = (
        output_dir.parent
        /
        "rejected"
    )

    removed_sets_dir = (
        output_dir.parent
        /
        "removed_sets"
    )

    (
        bad_sets,
        unresolved_records,

    ) = load_bad_sets(
        unresolved_path
    )

    print(
        "Unresolved evidence occurrences:",
        len(unresolved_records),
    )

    print(
        "Affected claim/set groups:",
        sum(
            len(indices)
            for indices
            in bad_sets.values()
        ),
    )

    report = {}

    clean_by_split = {}

    for split_name in SPLITS:

        print(
            f"\nCleaning {split_name}..."
        )

        input_path = (
            resolved_dir
            /
            f"{split_name}.jsonl"
        )

        if not input_path.exists():

            raise FileNotFoundError(
                f"Missing resolved dataset: "
                f"{input_path}"
            )

        records = load_jsonl(
            input_path
        )

        (
            clean_records,
            rejected_claims,
            removed_sets,
            stats,

        ) = clean_split(

            split_name,
            records,
            bad_sets,
        )

        write_jsonl(
            clean_records,
            output_dir
            /
            f"{split_name}.jsonl",
        )

        write_jsonl(
            rejected_claims,
            rejected_dir
            /
            f"{split_name}_no_complete_evidence.jsonl",
        )

        write_jsonl(
            removed_sets,
            removed_sets_dir
            /
            f"{split_name}_removed_sets.jsonl",
        )

        clean_by_split[
            split_name
        ] = clean_records

        report[
            split_name
        ] = stats

        print(
            "Input claims:",
            stats[
                "input_claims"
            ],
        )

        print(
            "Output claims:",
            stats[
                "output_claims"
            ],
        )

        print(
            "Rejected claims:",
            stats[
                "rejected_claims"
            ],
        )

        print(
            "Affected but kept:",
            stats[
                "affected_but_kept"
            ],
        )

        print(
            "Removed evidence sets:",
            stats[
                "removed_evidence_sets"
            ],
        )

        print(
            "True:",
            stats["true"],
        )

        print(
            "False:",
            stats["false"],
        )

    validate_cross_split_ids(
        clean_by_split
    )

    report["overall"] = {
        "input_claims":
            sum(
                report[split][
                    "input_claims"
                ]
                for split in SPLITS
            ),

        "output_claims":
            sum(
                report[split][
                    "output_claims"
                ]
                for split in SPLITS
            ),

        "rejected_claims":
            sum(
                report[split][
                    "rejected_claims"
                ]
                for split in SPLITS
            ),

        "removed_evidence_sets":
            sum(
                report[split][
                    "removed_evidence_sets"
                ]
                for split in SPLITS
            ),

        "unresolved_evidence_occurrences":
            len(
                unresolved_records
            ),
    }

    report_path = (
        output_dir.parent
        /
        "gold_cleaning_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nGold Evidence cleaning "
        "completed successfully."
    )

    print(
        "Output:",
        output_dir,
    )

    print(
        "Report:",
        report_path,
    )

if __name__ == "__main__":
    main()