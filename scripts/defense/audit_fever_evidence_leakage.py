from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("paper_test", "paper_dev", "train")
PRIORITY = {name: rank for rank, name in enumerate(SPLITS)}


def normalize_page(page: str) -> str:
    return unicodedata.normalize("NFC", page.strip())


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.strip().split()).casefold()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if line:
                yield line_number, json.loads(line)


def write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def get_claim_id(record: dict):
    return record.get(
        "claim_id",
        record.get("id"),
    )


def get_evidence_sets(record: dict) -> list[list[dict]]:
    raw = record.get("evidence_sets")

    if raw is None:
        raw = record.get("evidence")

    if raw is None:
        return []

    if not isinstance(raw, list):
        raise ValueError(
            "evidence_sets must be a list"
        )

    result = []

    for evidence_set in raw:
        if isinstance(evidence_set, list):
            items = evidence_set

        elif isinstance(evidence_set, dict):
            items = None

            for key in (
                "evidence",
                "items",
                "sentences",
            ):
                value = evidence_set.get(key)

                if isinstance(value, list):
                    items = value
                    break

            if items is None:
                raise ValueError(
                    "Unsupported evidence-set object schema"
                )

        else:
            raise ValueError(
                "Unsupported evidence-set schema"
            )

        if not all(
            isinstance(item, dict)
            for item in items
        ):
            raise ValueError(
                "Evidence items must be objects"
            )

        result.append(items)

    return result


def evidence_identity(item: dict):

    page = item.get("page")
    sentence_id = item.get("sentence_id")
    text = item.get("text")

    if (
        not isinstance(page, str)
        or not page.strip()
    ):
        raise ValueError("Missing page")

    if not isinstance(sentence_id, int):
        raise ValueError(
            "Invalid sentence_id"
        )

    if (
        not isinstance(text, str)
        or not text.strip()
    ):
        raise ValueError(
            "Missing evidence text"
        )

    page_nfc = normalize_page(page)

    ref_key = (
        f"{page_nfc}\u241f{sentence_id}"
    )

    normalized = normalize_text(text)

    text_hash = sha256_text(
        normalized
    )

    return (
        page_nfc,
        ref_key,
        text_hash,
        normalized,
    )


def add_presence(
    store: dict,
    key: str,
    split: str,
) -> None:

    store[key].add(split)


def owner_of(
    splits: set[str],
) -> str:

    return min(
        splits,
        key=lambda split: PRIORITY[split],
    )


def build_index(
    paths: dict[str, Path],
):

    page_presence = defaultdict(set)
    ref_presence = defaultdict(set)
    text_presence = defaultdict(set)

    text_examples = {}

    split_stats = {}
    schema_errors = []

    for split in SPLITS:

        path = paths[split]

        stats = Counter()

        pages = set()
        refs = set()
        texts = set()

        for (
            line_number,
            record,
        ) in read_jsonl(path):

            stats["claims"] += 1

            try:
                evidence_sets = (
                    get_evidence_sets(
                        record
                    )
                )

            except Exception as error:

                schema_errors.append(
                    {
                        "split":
                            split,

                        "line_number":
                            line_number,

                        "claim_id":
                            get_claim_id(
                                record
                            ),

                        "error":
                            str(error),
                    }
                )

                continue

            stats[
                "evidence_sets"
            ] += len(evidence_sets)

            for evidence_set in evidence_sets:

                stats[
                    "evidence_occurrences"
                ] += len(evidence_set)

                for item in evidence_set:

                    try:
                        (
                            page,
                            ref_key,
                            text_hash,
                            normalized,

                        ) = evidence_identity(
                            item
                        )

                    except Exception as error:

                        schema_errors.append(
                            {
                                "split":
                                    split,

                                "line_number":
                                    line_number,

                                "claim_id":
                                    get_claim_id(
                                        record
                                    ),

                                "error":
                                    str(error),
                            }
                        )

                        continue

                    pages.add(page)
                    refs.add(ref_key)
                    texts.add(text_hash)

                    add_presence(
                        page_presence,
                        page,
                        split,
                    )

                    add_presence(
                        ref_presence,
                        ref_key,
                        split,
                    )

                    add_presence(
                        text_presence,
                        text_hash,
                        split,
                    )

                    text_examples.setdefault(
                        text_hash,
                        normalized[:500],
                    )

        stats[
            "unique_pages"
        ] = len(pages)

        stats[
            "unique_evidence_refs"
        ] = len(refs)

        stats[
            "unique_normalized_texts"
        ] = len(texts)

        split_stats[
            split
        ] = dict(stats)

    return {
        "page_presence":
            page_presence,

        "ref_presence":
            ref_presence,

        "text_presence":
            text_presence,

        "text_examples":
            text_examples,

        "split_stats":
            split_stats,

        "schema_errors":
            schema_errors,
    }


def pairwise_overlap(
    presence: dict[str, set[str]],
) -> dict[str, int]:

    result = {}

    pairs = (
        (
            "paper_test",
            "paper_dev",
        ),
        (
            "paper_test",
            "train",
        ),
        (
            "paper_dev",
            "train",
        ),
    )

    for left, right in pairs:

        count = sum(
            left in splits
            and right in splits
            for splits
            in presence.values()
        )

        result[
            f"{left}__{right}"
        ] = count

    return result


def overlap_records(
    presence,
    entity_type,
    text_examples=None,
):

    for key in sorted(presence):

        splits = presence[key]

        if len(splits) <= 1:
            continue

        record = {
            "entity_type":
                entity_type,

            "key":
                key,

            "splits":
                sorted(
                    splits,
                    key=lambda split:
                        PRIORITY[split],
                ),

            "owner_split":
                owner_of(splits),
        }

        if text_examples is not None:

            record[
                "normalized_text_example"
            ] = text_examples.get(key)

        yield record


def build_owners(
    presence,
):

    return {
        key: owner_of(splits)
        for key, splits
        in presence.items()
    }


def audit_rejections(
    paths,
    index,
):

    page_owner = build_owners(
        index["page_presence"]
    )

    ref_owner = build_owners(
        index["ref_presence"]
    )

    text_owner = build_owners(
        index["text_presence"]
    )

    context_rejections = []
    sentence_rejections = []

    counters = {
        split: Counter()
        for split in SPLITS
    }

    for split in SPLITS:

        for (
            line_number,
            record,

        ) in read_jsonl(
            paths[split]
        ):

            claim_id = get_claim_id(
                record
            )

            try:
                evidence_sets = (
                    get_evidence_sets(
                        record
                    )
                )

            except Exception:
                continue

            for (
                set_index,
                evidence_set,

            ) in enumerate(
                evidence_sets
            ):

                context_reasons = set()
                context_refs = []

                valid_items = 0

                for item in evidence_set:

                    try:
                        (
                            page,
                            ref_key,
                            text_hash,
                            _,

                        ) = evidence_identity(
                            item
                        )

                    except Exception:
                        continue

                    valid_items += 1

                    context_refs.append(
                        {
                            "page":
                                page,

                            "sentence_id":
                                item[
                                    "sentence_id"
                                ],
                        }
                    )

                    reasons = []

                    if (
                        PRIORITY[
                            page_owner[
                                page
                            ]
                        ]
                        <
                        PRIORITY[split]
                    ):

                        reasons.append(
                            "higher_priority_page_overlap"
                        )

                    if (
                        PRIORITY[
                            ref_owner[
                                ref_key
                            ]
                        ]
                        <
                        PRIORITY[split]
                    ):

                        reasons.append(
                            "higher_priority_evidence_ref_overlap"
                        )

                    if (
                        PRIORITY[
                            text_owner[
                                text_hash
                            ]
                        ]
                        <
                        PRIORITY[split]
                    ):

                        reasons.append(
                            "higher_priority_normalized_text_overlap"
                        )

                    if reasons:

                        counters[
                            split
                        ][
                            "sentence_occurrences_rejected"
                        ] += 1

                        for reason in reasons:

                            counters[
                                split
                            ][reason] += 1

                            context_reasons.add(
                                reason
                            )

                        sentence_rejections.append(
                            {
                                "split":
                                    split,

                                "line_number":
                                    line_number,

                                "claim_id":
                                    claim_id,

                                "evidence_set_index":
                                    set_index,

                                "page":
                                    page,

                                "sentence_id":
                                    item[
                                        "sentence_id"
                                    ],

                                "normalized_text_sha256":
                                    text_hash,

                                "reasons":
                                    reasons,

                                "page_owner_split":
                                    page_owner[
                                        page
                                    ],

                                "ref_owner_split":
                                    ref_owner[
                                        ref_key
                                    ],

                                "text_owner_split":
                                    text_owner[
                                        text_hash
                                    ],
                            }
                        )

                counters[
                    split
                ][
                    "contexts_total"
                ] += 1

                if (
                    valid_items
                    != len(evidence_set)
                ):

                    context_reasons.add(
                        "invalid_or_incomplete_evidence_item"
                    )

                if context_reasons:

                    counters[
                        split
                    ][
                        "contexts_rejected"
                    ] += 1

                    context_rejections.append(
                        {
                            "split":
                                split,

                            "line_number":
                                line_number,

                            "claim_id":
                                claim_id,

                            "evidence_set_index":
                                set_index,

                            "evidence_refs":
                                context_refs,

                            "reasons":
                                sorted(
                                    context_reasons
                                ),
                        }
                    )

                else:

                    counters[
                        split
                    ][
                        "contexts_retained"
                    ] += 1

    return (
        counters,
        sentence_rejections,
        context_rejections,
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default=(
            "data/processed/fever/"
            "gold/clean"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/defense/"
            "audit/fever/3b1"
        ),
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    paths = {
        split:
            input_dir
            /
            f"{split}.jsonl"

        for split in SPLITS
    }

    missing = [
        str(path)
        for path
        in paths.values()
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing FEVER Gold Clean files:\n"
            +
            "\n".join(missing)
        )

    index = build_index(
        paths
    )

    (
        counters,
        sentence_rejections,
        context_rejections,

    ) = audit_rejections(
        paths,
        index,
    )

    page_pairwise = pairwise_overlap(
        index[
            "page_presence"
        ]
    )

    ref_pairwise = pairwise_overlap(
        index[
            "ref_presence"
        ]
    )

    text_pairwise = pairwise_overlap(
        index[
            "text_presence"
        ]
    )

    report = {
        "policy": {
            "split_priority":
                list(SPLITS),

            "page_normalization":
                "Unicode NFC",

            "text_normalization_for_audit_only":
                (
                    "Unicode NFKC + strip + "
                    "collapse_whitespace + casefold"
                ),

            "page_disjoint":
                True,

            "context_policy":
                (
                    "reject_complete_context_if_any_"
                    "evidence_item_is_leaky"
                ),

            "paper_test_locked":
                True,
        },

        "split_stats":
            index[
                "split_stats"
            ],

        "pairwise_overlap": {
            "page":
                page_pairwise,

            "evidence_ref":
                ref_pairwise,

            "normalized_text":
                text_pairwise,
        },

        "proposed_rejection_stats": {
            split:
                dict(
                    counters[split]
                )

            for split in SPLITS
        },

        "schema_error_count":
            len(
                index[
                    "schema_errors"
                ]
            ),

        "overlap_entity_counts": {
            "pages_in_multiple_splits":
                sum(
                    len(splits) > 1
                    for splits
                    in index[
                        "page_presence"
                    ].values()
                ),

            "evidence_refs_in_multiple_splits":
                sum(
                    len(splits) > 1
                    for splits
                    in index[
                        "ref_presence"
                    ].values()
                ),

            "normalized_texts_in_multiple_splits":
                sum(
                    len(splits) > 1
                    for splits
                    in index[
                        "text_presence"
                    ].values()
                ),
        },
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        /
        "leakage_report.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_jsonl(
        overlap_records(
            index[
                "page_presence"
            ],
            "page",
        ),
        output_dir
        /
        "page_overlap.jsonl",
    )

    write_jsonl(
        overlap_records(
            index[
                "ref_presence"
            ],
            "evidence_ref",
        ),
        output_dir
        /
        "evidence_ref_overlap.jsonl",
    )

    write_jsonl(
        overlap_records(
            index[
                "text_presence"
            ],
            "normalized_text",
            index[
                "text_examples"
            ],
        ),
        output_dir
        /
        "normalized_text_overlap.jsonl",
    )

    write_jsonl(
        sentence_rejections,
        output_dir
        /
        "proposed_sentence_rejections.jsonl",
    )

    write_jsonl(
        context_rejections,
        output_dir
        /
        "proposed_context_rejections.jsonl",
    )

    write_jsonl(
        index[
            "schema_errors"
        ],
        output_dir
        /
        "schema_errors.jsonl",
    )

    print(
        "\n================================"
    )

    print(
        "FEVER Evidence Leakage Audit"
    )

    print(
        "================================"
    )

    for split in SPLITS:

        stats = index[
            "split_stats"
        ][split]

        print(
            f"\n{split}"
        )

        print(
            "  claims:",
            stats.get(
                "claims",
                0,
            ),
        )

        print(
            "  evidence sets:",
            stats.get(
                "evidence_sets",
                0,
            ),
        )

        print(
            "  evidence occurrences:",
            stats.get(
                "evidence_occurrences",
                0,
            ),
        )

        print(
            "  unique pages:",
            stats.get(
                "unique_pages",
                0,
            ),
        )

        print(
            "  unique refs:",
            stats.get(
                "unique_evidence_refs",
                0,
            ),
        )

        print(
            "  unique normalized texts:",
            stats.get(
                "unique_normalized_texts",
                0,
            ),
        )

    print(
        "\nPairwise page overlap:"
    )

    for pair, count in (
        page_pairwise.items()
    ):

        print(
            f"  {pair}: {count}"
        )

    print(
        "\nPairwise evidence-ref overlap:"
    )

    for pair, count in (
        ref_pairwise.items()
    ):

        print(
            f"  {pair}: {count}"
        )

    print(
        "\nPairwise normalized-text overlap:"
    )

    for pair, count in (
        text_pairwise.items()
    ):

        print(
            f"  {pair}: {count}"
        )

    print(
        "\nProposed context rejection:"
    )

    for split in SPLITS:

        stats = counters[
            split
        ]

        print(
            f"  {split}: "
            f"{stats.get('contexts_rejected', 0)} / "
            f"{stats.get('contexts_total', 0)}"
        )

    print(
        "\nSchema errors:",
        len(
            index[
                "schema_errors"
            ]
        ),
    )

    print(
        "Report:",
        output_dir
        /
        "leakage_report.json",
    )


if __name__ == "__main__":
    main()