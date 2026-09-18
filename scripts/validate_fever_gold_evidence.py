from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path


SPLITS = (
    "train",
    "paper_dev",
    "paper_test",
)

LABEL_MAP = {
    "SUPPORTS": "True",
    "REFUTES": "False",
}

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

    text = " ".join(
        text.strip().split()
    )

    return text.casefold()

def validate_split(
    split_name: str,
    gold_records: list[dict],
    clean_audit_records: list[dict],
    rejected_records: list[dict],
):

    errors = []

    warnings = []

    clean_by_id = {
        int(record["claim_id"]): record
        for record in clean_audit_records
    }

    gold_by_id = {}

    claim_counter = Counter()

    normalized_counter = Counter()

    evidence_set_count = 0

    evidence_sentence_count = 0

    duplicate_evidence_refs = 0

    true_count = 0

    false_count = 0

    for record_index, record in enumerate(
        gold_records
    ):

        required_fields = {
            "claim_id",
            "claim",
            "label",
            "fever_label",
            "evidence_sets",
            "split",
        }

        missing_fields = (
            required_fields
            -
            set(record.keys())
        )

        if missing_fields:

            errors.append(
                {
                    "type":
                        "missing_claim_fields",

                    "record_index":
                        record_index,

                    "missing":
                        sorted(
                            missing_fields
                        ),
                }
            )

            continue

        claim_id = int(
            record["claim_id"]
        )

        claim = record[
            "claim"
        ]

        label = record[
            "label"
        ]

        fever_label = record[
            "fever_label"
        ]

        if claim_id in gold_by_id:

            errors.append(
                {
                    "type":
                        "duplicate_claim_id",

                    "claim_id":
                        claim_id,
                }
            )

        gold_by_id[
            claim_id
        ] = record

        if record["split"] != split_name:

            errors.append(
                {
                    "type":
                        "wrong_split",

                    "claim_id":
                        claim_id,

                    "value":
                        record["split"],
                }
            )

        if label not in {
            "True",
            "False",
        }:

            errors.append(
                {
                    "type":
                        "invalid_binary_label",

                    "claim_id":
                        claim_id,

                    "label":
                        label,
                }
            )

        if fever_label not in LABEL_MAP:

            errors.append(
                {
                    "type":
                        "invalid_fever_label",

                    "claim_id":
                        claim_id,

                    "fever_label":
                        fever_label,
                }
            )

        else:

            expected_label = (
                LABEL_MAP[
                    fever_label
                ]
            )

            if label != expected_label:

                errors.append(
                    {
                        "type":
                            "label_mapping_mismatch",

                        "claim_id":
                            claim_id,

                        "fever_label":
                            fever_label,

                        "label":
                            label,
                    }
                )

        if label == "True":
            true_count += 1

        elif label == "False":
            false_count += 1

        if claim_id not in clean_by_id:

            errors.append(
                {
                    "type":
                        "claim_not_in_clean_binary",

                    "claim_id":
                        claim_id,
                }
            )

        else:

            clean_record = clean_by_id[
                claim_id
            ]

            if claim != clean_record[
                "claim"
            ]:

                errors.append(
                    {
                        "type":
                            "claim_text_mismatch",

                        "claim_id":
                            claim_id,
                    }
                )

            if label != clean_record[
                "label"
            ]:

                errors.append(
                    {
                        "type":
                            "binary_label_mismatch",

                        "claim_id":
                            claim_id,
                    }
                )

        # ----------------------------------------------------
        # Duplicate text analysis
        # ----------------------------------------------------

        claim_counter[
            claim
        ] += 1

        normalized_counter[
            normalize_claim(
                claim
            )
        ] += 1

        # ====================================================
        # Evidence sets
        # ====================================================

        evidence_sets = record[
            "evidence_sets"
        ]

        if (
            not isinstance(
                evidence_sets,
                list,
            )
            or
            not evidence_sets
        ):

            errors.append(
                {
                    "type":
                        "no_evidence_sets",

                    "claim_id":
                        claim_id,
                }
            )

            continue

        evidence_set_count += len(
            evidence_sets
        )

        for set_index, evidence_set \
                in enumerate(
                    evidence_sets
                ):

            if (
                not isinstance(
                    evidence_set,
                    list,
                )
                or
                not evidence_set
            ):

                errors.append(
                    {
                        "type":
                            "empty_evidence_set",

                        "claim_id":
                            claim_id,

                        "evidence_set_index":
                            set_index,
                    }
                )

                continue

            seen_refs = set()

            # ================================================
            # Evidence items
            # ================================================

            for evidence_index, evidence \
                    in enumerate(
                        evidence_set
                    ):

                evidence_sentence_count += 1

                required_evidence_fields = {
                    "annotation_id",
                    "evidence_id",
                    "page",
                    "sentence_id",
                    "text",
                }

                missing = (
                    required_evidence_fields
                    -
                    set(evidence.keys())
                )

                if missing:

                    errors.append(
                        {
                            "type":
                                "missing_evidence_fields",

                            "claim_id":
                                claim_id,

                            "evidence_set_index":
                                set_index,

                            "evidence_index":
                                evidence_index,

                            "missing":
                                sorted(
                                    missing
                                ),
                        }
                    )

                    continue

                page = evidence[
                    "page"
                ]

                sentence_id = evidence[
                    "sentence_id"
                ]

                text = evidence[
                    "text"
                ]

                # --------------------------------------------
                # Page
                # --------------------------------------------

                if (
                    not isinstance(
                        page,
                        str,
                    )
                    or
                    not page.strip()
                ):

                    errors.append(
                        {
                            "type":
                                "invalid_page",

                            "claim_id":
                                claim_id,

                            "evidence_set_index":
                                set_index,

                            "evidence_index":
                                evidence_index,
                        }
                    )

                # --------------------------------------------
                # sentence_id
                # --------------------------------------------

                if (
                    not isinstance(
                        sentence_id,
                        int,
                    )
                    or
                    sentence_id < 0
                ):

                    errors.append(
                        {
                            "type":
                                "invalid_sentence_id",

                            "claim_id":
                                claim_id,

                            "value":
                                sentence_id,
                        }
                    )

                # --------------------------------------------
                # Evidence text
                # --------------------------------------------

                if (
                    not isinstance(
                        text,
                        str,
                    )
                    or
                    not text.strip()
                ):

                    errors.append(
                        {
                            "type":
                                "empty_evidence_text",

                            "claim_id":
                                claim_id,

                            "page":
                                page,

                            "sentence_id":
                                sentence_id,
                        }
                    )

                # --------------------------------------------
                # Duplicate reference inside same evidence set
                # --------------------------------------------

                ref = (
                    page,
                    sentence_id,
                )

                if ref in seen_refs:

                    duplicate_evidence_refs += 1

                    warnings.append(
                        {
                            "type":
                                "duplicate_evidence_reference",

                            "claim_id":
                                claim_id,

                            "evidence_set_index":
                                set_index,

                            "page":
                                page,

                            "sentence_id":
                                sentence_id,
                        }
                    )

                seen_refs.add(
                    ref
                )

    # ========================================================
    # Claim duplicates
    # ========================================================

    exact_duplicate_groups = sum(
        count > 1
        for count in claim_counter.values()
    )

    normalized_duplicate_groups = sum(
        count > 1
        for count
        in normalized_counter.values()
    )

    if exact_duplicate_groups:

        errors.append(
            {
                "type":
                    "exact_duplicate_claim_groups",

                "count":
                    exact_duplicate_groups,
            }
        )

    if normalized_duplicate_groups:

        errors.append(
            {
                "type":
                    "normalized_duplicate_claim_groups",

                "count":
                    normalized_duplicate_groups,
            }
        )

    # ========================================================
    # Verify rejected claims
    # ========================================================

    clean_ids = set(
        clean_by_id
    )

    gold_ids = set(
        gold_by_id
    )

    missing_from_gold = (
        clean_ids
        -
        gold_ids
    )

    rejected_ids = {
        int(record["claim_id"])
        for record in rejected_records
    }

    missing_not_rejected = (
        missing_from_gold
        -
        rejected_ids
    )

    rejected_but_present = (
        rejected_ids
        &
        gold_ids
    )

    if missing_not_rejected:

        errors.append(
            {
                "type":
                    "missing_claim_not_documented_as_rejected",

                "count":
                    len(
                        missing_not_rejected
                    ),

                "examples":
                    sorted(
                        missing_not_rejected
                    )[:10],
            }
        )

    if rejected_but_present:

        errors.append(
            {
                "type":
                    "rejected_claim_present_in_gold",

                "count":
                    len(
                        rejected_but_present
                    ),

                "examples":
                    sorted(
                        rejected_but_present
                    )[:10],
            }
        )

    if rejected_ids != missing_from_gold:

        errors.append(
            {
                "type":
                    "rejected_set_mismatch",

                "clean_minus_gold":
                    len(
                        missing_from_gold
                    ),

                "rejected_file":
                    len(
                        rejected_ids
                    ),
            }
        )

    stats = {
        "claims":
            len(
                gold_records
            ),

        "true":
            true_count,

        "false":
            false_count,

        "evidence_sets":
            evidence_set_count,

        "evidence_sentences":
            evidence_sentence_count,

        "exact_duplicate_claim_groups":
            exact_duplicate_groups,

        "normalized_duplicate_claim_groups":
            normalized_duplicate_groups,

        "duplicate_evidence_refs":
            duplicate_evidence_refs,

        "rejected_claims":
            len(
                rejected_ids
            ),

        "errors":
            len(
                errors
            ),

        "warnings":
            len(
                warnings
            ),
    }

    return (
        stats,
        errors,
        warnings,
        gold_by_id,
    )


# ============================================================
# Cross-split validation
# ============================================================

def validate_cross_split(
    gold_by_split,
):

    results = {}

    errors = []

    for i, split_a in enumerate(
        SPLITS
    ):

        for split_b in SPLITS[
            i + 1:
        ]:

            records_a = (
                gold_by_split[
                    split_a
                ]
            )

            records_b = (
                gold_by_split[
                    split_b
                ]
            )

            ids_a = set(
                records_a
            )

            ids_b = set(
                records_b
            )

            id_overlap = (
                ids_a
                &
                ids_b
            )

            normalized_a = {
                normalize_claim(
                    record["claim"]
                )
                for record
                in records_a.values()
            }

            normalized_b = {
                normalize_claim(
                    record["claim"]
                )
                for record
                in records_b.values()
            }

            text_overlap = (
                normalized_a
                &
                normalized_b
            )

            key = (
                f"{split_a}__"
                f"{split_b}"
            )

            results[key] = {
                "claim_id_overlap":
                    len(
                        id_overlap
                    ),

                "normalized_claim_overlap":
                    len(
                        text_overlap
                    ),
            }

            if id_overlap:

                errors.append(
                    {
                        "type":
                            "cross_split_claim_id_overlap",

                        "splits":
                            key,

                        "count":
                            len(
                                id_overlap
                            ),
                    }
                )

            if text_overlap:

                errors.append(
                    {
                        "type":
                            "cross_split_claim_text_overlap",

                        "splits":
                            key,

                        "count":
                            len(
                                text_overlap
                            ),
                    }
                )

    return (
        results,
        errors,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold-dir",
        default=(
            "data/processed/fever/"
            "gold/clean"
        ),
    )

    parser.add_argument(
        "--clean-audit-dir",
        default=(
            "data/processed/fever/"
            "clean/audit"
        ),
    )

    parser.add_argument(
        "--rejected-dir",
        default=(
            "data/processed/fever/"
            "gold/rejected"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "data/processed/fever/"
            "gold/gold_validation_report.json"
        ),
    )

    args = parser.parse_args()

    gold_dir = Path(
        args.gold_dir
    )

    clean_audit_dir = Path(
        args.clean_audit_dir
    )

    rejected_dir = Path(
        args.rejected_dir
    )

    report_path = Path(
        args.report
    )

    report = {
        "splits": {},
    }

    all_errors = []

    all_warnings = []

    gold_by_split = {}

    # ========================================================
    # Split validation
    # ========================================================

    for split_name in SPLITS:

        print(
            f"\nValidating "
            f"{split_name}..."
        )

        gold_records = load_jsonl(
            gold_dir
            /
            f"{split_name}.jsonl"
        )

        clean_audit_records = load_jsonl(
            clean_audit_dir
            /
            f"{split_name}.jsonl"
        )

        rejected_records = load_jsonl(
            rejected_dir
            /
            (
                f"{split_name}"
                "_no_complete_evidence.jsonl"
            )
        )

        (
            stats,
            errors,
            warnings,
            gold_by_id,

        ) = validate_split(

            split_name,
            gold_records,
            clean_audit_records,
            rejected_records,
        )

        gold_by_split[
            split_name
        ] = gold_by_id

        report["splits"][
            split_name
        ] = stats

        for error in errors:

            error[
                "split"
            ] = split_name

        for warning in warnings:

            warning[
                "split"
            ] = split_name

        all_errors.extend(
            errors
        )

        all_warnings.extend(
            warnings
        )

        print(
            "Claims:",
            stats["claims"],
        )

        print(
            "True:",
            stats["true"],
        )

        print(
            "False:",
            stats["false"],
        )

        print(
            "Evidence sets:",
            stats[
                "evidence_sets"
            ],
        )

        print(
            "Evidence sentences:",
            stats[
                "evidence_sentences"
            ],
        )

        print(
            "Duplicate claims:",
            stats[
                "normalized_duplicate_claim_groups"
            ],
        )

        print(
            "Errors:",
            stats["errors"],
        )

        print(
            "Warnings:",
            stats["warnings"],
        )

    # ========================================================
    # Cross-split validation
    # ========================================================

    print(
        "\nChecking cross-split leakage..."
    )

    (
        cross_split,
        cross_errors,

    ) = validate_cross_split(
        gold_by_split
    )

    all_errors.extend(
        cross_errors
    )

    report[
        "cross_split"
    ] = cross_split

    for key, values \
            in cross_split.items():

        print(
            f"\n{key}"
        )

        print(
            "  claim_id overlap:",
            values[
                "claim_id_overlap"
            ],
        )

        print(
            "  normalized claim overlap:",
            values[
                "normalized_claim_overlap"
            ],
        )

    # ========================================================
    # Final report
    # ========================================================

    report[
        "summary"
    ] = {
        "total_errors":
            len(
                all_errors
            ),

        "total_warnings":
            len(
                all_warnings
            ),

        "status":
            (
                "PASS"
                if not all_errors
                else "FAIL"
            ),
    }

    report[
        "errors"
    ] = all_errors

    report[
        "warnings"
    ] = all_warnings

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

    print(
        "\n================================"
    )

    print(
        "Gold Evidence Validation"
    )

    print(
        "Status:",
        report[
            "summary"
        ][
            "status"
        ],
    )

    print(
        "Errors:",
        len(
            all_errors
        ),
    )

    print(
        "Warnings:",
        len(
            all_warnings
        ),
    )

    print(
        "Report:",
        report_path,
    )

    if all_errors:

        raise RuntimeError(
            "Gold Evidence validation "
            "FAILED. Check validation report."
        )

if __name__ == "__main__":
    main()