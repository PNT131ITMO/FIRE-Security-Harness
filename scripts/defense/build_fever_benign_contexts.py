from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("paper_test", "paper_dev", "train")
PRIORITY = {
    split: rank
    for rank, split in enumerate(SPLITS)
}


def normalize_page(page: str) -> str:
    return unicodedata.normalize(
        "NFC",
        page.strip(),
    )


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


def read_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if line:
                yield (
                    line_number,
                    json.loads(line),
                )


def write_jsonl(
    records,
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


def get_claim_id(record: dict):
    return record.get(
        "claim_id",
        record.get("id"),
    )


def get_fever_label(record: dict):
    return record.get(
        "fever_label",
        record.get("label"),
    )


def get_evidence_sets(
    record: dict,
) -> list[list[dict]]:

    value = record.get(
        "evidence_sets"
    )

    if value is None:
        value = record.get(
            "evidence"
        )

    if not isinstance(value, list):
        raise ValueError(
            "Evidence sets are missing"
        )

    result = []

    for evidence_set in value:

        if isinstance(
            evidence_set,
            list,
        ):

            items = evidence_set

        elif isinstance(
            evidence_set,
            dict,
        ):

            items = None

            for key in (
                "evidence",
                "items",
                "sentences",
            ):

                candidate = (
                    evidence_set.get(key)
                )

                if isinstance(
                    candidate,
                    list,
                ):

                    items = candidate
                    break

            if items is None:
                raise ValueError(
                    "Unsupported evidence-set schema"
                )

        else:

            raise ValueError(
                "Unsupported evidence-set schema"
            )

        result.append(items)

    return result


def parse_evidence(
    item: dict,
) -> dict:

    page = item.get("page")
    sentence_id = item.get(
        "sentence_id"
    )
    text = item.get("text")

    if (
        not isinstance(page, str)
        or not page.strip()
    ):

        raise ValueError(
            "Missing page"
        )

    if not isinstance(
        sentence_id,
        int,
    ):

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

    canonical_page = (
        item.get(
            "canonical_page"
        )
    )

    if (
        not isinstance(
            canonical_page,
            str,
        )
        or not canonical_page.strip()
    ):

        canonical_page = (
            normalize_page(page)
        )

    else:

        canonical_page = (
            normalize_page(
                canonical_page
            )
        )

    normalized = normalize_text(
        text
    )

    ref_identity = (
        f"{canonical_page}\u241f"
        f"{sentence_id}"
    )

    return {
        "page":
            page,

        "canonical_page":
            canonical_page,

        "sentence_id":
            sentence_id,

        "text":
            text,

        "normalized_text":
            normalized,

        "normalized_text_sha256":
            sha256_text(normalized),

        "ref_identity":
            ref_identity,
    }


def owner_of(
    splits: set[str],
) -> str:

    return min(
        splits,
        key=lambda split:
            PRIORITY[split],
    )


def build_ownership(
    paths: dict[str, Path],
):

    pages = defaultdict(set)
    refs = defaultdict(set)
    texts = defaultdict(set)

    errors = []

    for split in SPLITS:

        for (
            line_number,
            record,

        ) in read_jsonl(
            paths[split]
        ):

            try:

                evidence_sets = (
                    get_evidence_sets(
                        record
                    )
                )

            except Exception as error:

                errors.append(
                    {
                        "split":
                            split,

                        "line_number":
                            line_number,

                        "claim_id":
                            get_claim_id(
                                record
                            ),

                        "reason":
                            str(error),
                    }
                )

                continue

            for evidence_set in evidence_sets:

                for item in evidence_set:

                    try:

                        evidence = (
                            parse_evidence(
                                item
                            )
                        )

                    except Exception as error:

                        errors.append(
                            {
                                "split":
                                    split,

                                "line_number":
                                    line_number,

                                "claim_id":
                                    get_claim_id(
                                        record
                                    ),

                                "reason":
                                    str(error),
                            }
                        )

                        continue

                    pages[
                        evidence[
                            "canonical_page"
                        ]
                    ].add(split)

                    refs[
                        evidence[
                            "ref_identity"
                        ]
                    ].add(split)

                    texts[
                        evidence[
                            "normalized_text_sha256"
                        ]
                    ].add(split)

    return {
        "page": {
            key: owner_of(value)
            for key, value
            in pages.items()
        },

        "ref": {
            key: owner_of(value)
            for key, value
            in refs.items()
        },

        "text": {
            key: owner_of(value)
            for key, value
            in texts.items()
        },

        "errors":
            errors,
    }


def leakage_reasons(
    evidence: dict,
    split: str,
    ownership: dict,
) -> list[str]:

    reasons = []

    page_owner = ownership[
        "page"
    ][
        evidence[
            "canonical_page"
        ]
    ]

    ref_owner = ownership[
        "ref"
    ][
        evidence[
            "ref_identity"
        ]
    ]

    text_owner = ownership[
        "text"
    ][
        evidence[
            "normalized_text_sha256"
        ]
    ]

    if (
        PRIORITY[page_owner]
        <
        PRIORITY[split]
    ):

        reasons.append(
            "higher_priority_page_overlap"
        )

    if (
        PRIORITY[ref_owner]
        <
        PRIORITY[split]
    ):

        reasons.append(
            "higher_priority_evidence_ref_overlap"
        )

    if (
        PRIORITY[text_owner]
        <
        PRIORITY[split]
    ):

        reasons.append(
            "higher_priority_normalized_text_overlap"
        )

    return reasons


def sentence_identity(
    evidence: dict,
) -> str:

    return (
        "sentence\u241f"
        f"{evidence['canonical_page']}"
        "\u241f"
        f"{evidence['sentence_id']}"
    )


def context_identity(
    evidences: list[dict],
) -> str:

    ordered = sorted(
        evidences,
        key=lambda evidence: (
            evidence["canonical_page"],
            evidence["sentence_id"],
        ),
    )

    refs = [
        evidence["ref_identity"]
        for evidence in ordered
    ]

    return (
        "context\u241f"
        +
        "\u241e".join(refs)
    )


def base_metadata(
    granularity: str,
    base_hash: str,
    text: str,
    split: str,
) -> dict:

    return {
        "sample_id":
            f"fever_{granularity}_{base_hash}",

        "label":
            "benign",

        "source":
            "fever",

        "original_source":
            "FEVER",

        "upstream_split":
            split,

        "locked":
            split == "paper_test",

        "granularity":
            granularity,

        "base_text_id":
            (
                f"fever:{granularity}:"
                f"{base_hash}"
            ),

        "base_text_sha256":
            base_hash,

        "text":
            text,

        "text_sha256":
            sha256_text(text),

        "normalized_text_sha256":
            sha256_text(
                normalize_text(text)
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
            "en",

        "synthetic":
            False,

        "split":
            split,
    }


def build_sentence_sample(
    evidence: dict,
    split: str,
    claim_id,
    fever_label,
) -> dict:

    identity = sentence_identity(
        evidence
    )

    base_hash = sha256_text(
        identity
    )

    sample = base_metadata(
        "sentence",
        base_hash,
        evidence["text"],
        split,
    )

    sample.update(
        {
            "claim_ids": set(
                [claim_id]
                if claim_id is not None
                else []
            ),

            "fever_labels": set(
                [str(fever_label)]
                if fever_label is not None
                else []
            ),

            "page":
                evidence["page"],

            "canonical_page":
                evidence[
                    "canonical_page"
                ],

            "original_pages": {
                evidence["page"]
            },

            "sentence_id":
                evidence[
                    "sentence_id"
                ],

            "evidence_refs": [
                {
                    "page":
                        evidence["page"],

                    "canonical_page":
                        evidence[
                            "canonical_page"
                        ],

                    "sentence_id":
                        evidence[
                            "sentence_id"
                        ],
                }
            ],

            "context_size": 1,

            "context_complete": None,
        }
    )

    return sample

def build_context_sample(
    evidences: list[dict],
    split: str,
    claim_id,
    fever_label,
) -> dict:

    ordered = sorted(
        evidences,
        key=lambda evidence: (
            evidence[
                "canonical_page"
            ],
            evidence[
                "sentence_id"
            ],
        ),
    )

    identity = context_identity(
        ordered
    )

    base_hash = sha256_text(
        identity
    )

    text = "\n".join(
        evidence["text"]
        for evidence
        in ordered
    )

    sample = base_metadata(
        "context",
        base_hash,
        text,
        split,
    )

    sample.update(
        {
            "claim_ids": set(
                [claim_id]
                if claim_id is not None
                else []
            ),

            "fever_labels": set(
                [str(fever_label)]
                if fever_label is not None
                else []
            ),

            "page": None,

            "canonical_page": None,

            "sentence_id": None,

            "evidence_refs": [
                {
                    "page":
                        evidence["page"],

                    "canonical_page":
                        evidence[
                            "canonical_page"
                        ],

                    "sentence_id":
                        evidence[
                            "sentence_id"
                        ],
                }

                for evidence
                in ordered
            ],

            "context_size":
                len(ordered),

            "context_complete": True,
        }
    )

    return sample

def merge_provenance(
    target: dict,
    source: dict,
) -> None:

    target[
        "claim_ids"
    ].update(
        source[
            "claim_ids"
        ]
    )

    target[
        "fever_labels"
    ].update(
        source[
            "fever_labels"
        ]
    )

    if (
        target[
            "granularity"
        ]
        == "sentence"
    ):

        target[
            "original_pages"
        ].update(
            source[
                "original_pages"
            ]
        )

def serialize_sample(
    sample: dict,
) -> dict:

    result = dict(sample)

    claim_ids = sorted(
        set(
            result.get(
                "claim_ids",
                []
            )
        ),
        key=str,
    )

    result["claim_ids"] = claim_ids

    result["primary_claim_id"] = (
        claim_ids[0]
        if claim_ids
        else None
    )

    result["fever_labels"] = sorted(
        set(
            result.get(
                "fever_labels",
                []
            )
        )
    )

    if "original_pages" in result:

        result["original_pages"] = sorted(
            set(
                result[
                    "original_pages"
                ]
            )
        )

    result.pop(
        "claim_id",
        None,
    )

    return result

def build_dataset(
    paths: dict[str, Path],
    ownership: dict,
):

    sentence_samples = {
        split: {}
        for split in SPLITS
    }

    context_samples = {
        split: {}
        for split in SPLITS
    }

    rejected_sentences = {}
    rejected_contexts = {}
    invalid_contexts = []

    stats = {
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

            fever_label = (
                get_fever_label(
                    record
                )
            )

            try:

                evidence_sets = (
                    get_evidence_sets(
                        record
                    )
                )

            except Exception as error:

                invalid_contexts.append(
                    {
                        "split":
                            split,

                        "line_number":
                            line_number,

                        "claim_id":
                            claim_id,

                        "reason":
                            str(error),
                    }
                )

                continue

            for (
                evidence_set_index,
                evidence_set,

            ) in enumerate(
                evidence_sets
            ):

                stats[
                    split
                ][
                    "input_context_occurrences"
                ] += 1

                parsed = []
                context_reasons = set()
                context_valid = True

                for item in evidence_set:

                    try:

                        evidence = (
                            parse_evidence(
                                item
                            )
                        )

                    except Exception as error:

                        context_valid = False

                        context_reasons.add(
                            "invalid_evidence_item"
                        )

                        invalid_contexts.append(
                            {
                                "split":
                                    split,

                                "line_number":
                                    line_number,

                                "claim_id":
                                    claim_id,

                                "evidence_set_index":
                                    evidence_set_index,

                                "reason":
                                    str(error),
                            }
                        )

                        continue

                    parsed.append(
                        evidence
                    )

                    reasons = (
                        leakage_reasons(
                            evidence,
                            split,
                            ownership,
                        )
                    )

                    if reasons:

                        context_reasons.update(
                            reasons
                        )

                        identity = (
                            sentence_identity(
                                evidence
                            )
                        )

                        rejected_id = (
                            sha256_text(
                                identity
                            )
                        )

                        key = (
                            split,
                            rejected_id,
                        )

                        if (
                            key
                            not in rejected_sentences
                        ):

                            rejected_sentences[
                                key
                            ] = {
                                "sample_id":
                                    (
                                        "fever_sentence_"
                                        f"{rejected_id}"
                                    ),

                                "split":
                                    split,

                                "claim_ids":
                                    set(),

                                "page":
                                    evidence[
                                        "page"
                                    ],

                                "canonical_page":
                                    evidence[
                                        "canonical_page"
                                    ],

                                "sentence_id":
                                    evidence[
                                        "sentence_id"
                                    ],

                                "reasons":
                                    set(),
                            }

                        if claim_id is not None:

                            rejected_sentences[
                                key
                            ][
                                "claim_ids"
                            ].add(
                                claim_id
                            )

                        rejected_sentences[
                            key
                        ][
                            "reasons"
                        ].update(
                            reasons
                        )

                    else:

                        sample = (
                            build_sentence_sample(
                                evidence,
                                split,
                                claim_id,
                                fever_label,
                            )
                        )

                        sample_id = (
                            sample[
                                "sample_id"
                            ]
                        )

                        existing = (
                            sentence_samples[
                                split
                            ].get(
                                sample_id
                            )
                        )

                        if existing is None:

                            sentence_samples[
                                split
                            ][
                                sample_id
                            ] = sample

                        else:

                            merge_provenance(
                                existing,
                                sample,
                            )

                if (
                    len(parsed)
                    != len(evidence_set)
                ):

                    context_valid = False

                    context_reasons.add(
                        "incomplete_context"
                    )

                ref_ids = [
                    evidence[
                        "ref_identity"
                    ]
                    for evidence
                    in parsed
                ]

                if (
                    len(ref_ids)
                    != len(set(ref_ids))
                ):

                    context_valid = False

                    context_reasons.add(
                        "duplicate_evidence_ref_within_context"
                    )

                if not parsed:

                    context_valid = False

                    context_reasons.add(
                        "empty_context"
                    )

                if (
                    not context_valid
                    or context_reasons
                ):

                    if parsed:

                        rejected_hash = (
                            sha256_text(
                                context_identity(
                                    parsed
                                )
                            )
                        )

                    else:

                        rejected_hash = (
                            sha256_text(
                                (
                                    f"{split}|"
                                    f"{line_number}|"
                                    f"{evidence_set_index}"
                                )
                            )
                        )

                    key = (
                        split,
                        rejected_hash,
                    )

                    if (
                        key
                        not in rejected_contexts
                    ):

                        rejected_contexts[
                            key
                        ] = {
                            "sample_id":
                                (
                                    "fever_context_"
                                    f"{rejected_hash}"
                                ),

                            "split":
                                split,

                            "claim_ids":
                                set(),

                            "evidence_refs": [
                                {
                                    "page":
                                        evidence[
                                            "page"
                                        ],

                                    "canonical_page":
                                        evidence[
                                            "canonical_page"
                                        ],

                                    "sentence_id":
                                        evidence[
                                            "sentence_id"
                                        ],
                                }

                                for evidence
                                in parsed
                            ],

                            "reasons":
                                set(),
                        }

                    if claim_id is not None:

                        rejected_contexts[
                            key
                        ][
                            "claim_ids"
                        ].add(
                            claim_id
                        )

                    rejected_contexts[
                        key
                    ][
                        "reasons"
                    ].update(
                        context_reasons
                    )

                    stats[
                        split
                    ][
                        "rejected_context_occurrences"
                    ] += 1

                    continue

                sample = (
                    build_context_sample(
                        parsed,
                        split,
                        claim_id,
                        fever_label,
                    )
                )

                sample_id = (
                    sample[
                        "sample_id"
                    ]
                )

                existing = (
                    context_samples[
                        split
                    ].get(
                        sample_id
                    )
                )

                if existing is None:

                    context_samples[
                        split
                    ][
                        sample_id
                    ] = sample

                else:

                    merge_provenance(
                        existing,
                        sample,
                    )

    return {
        "sentence_samples":
            sentence_samples,

        "context_samples":
            context_samples,

        "rejected_sentences":
            rejected_sentences,

        "rejected_contexts":
            rejected_contexts,

        "invalid_contexts":
            invalid_contexts,

        "stats":
            stats,
    }


def serialize_rejected(
    records: dict,
):

    result = []

    for record in records.values():

        item = dict(record)

        item[
            "claim_ids"
        ] = sorted(
            item[
                "claim_ids"
            ],
            key=str,
        )

        item[
            "reasons"
        ] = sorted(
            item[
                "reasons"
            ]
        )

        result.append(item)

    return sorted(
        result,
        key=lambda item:
            item["sample_id"],
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
            "sources/fever"
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

    for path in paths.values():

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    print(
        "Building split ownership..."
    )

    ownership = build_ownership(
        paths
    )

    if ownership["errors"]:

        print(
            "Ownership schema errors:",
            len(
                ownership[
                    "errors"
                ]
            ),
        )

    print(
        "Building FEVER benign contexts..."
    )

    result = build_dataset(
        paths,
        ownership,
    )

    report = {
        "policy": {
            "priority":
                list(SPLITS),

            "page_disjoint":
                True,

            "evidence_ref_disjoint":
                True,

            "normalized_text_disjoint":
                True,

            "page_normalization":
                "Unicode NFC",

            "text_normalization":
                (
                    "NFKC + strip + "
                    "collapse_whitespace + casefold"
                ),

            "paper_test_locked":
                True,

            "defense_label":
                "benign",
        },

        "splits": {},
    }

    for split in SPLITS:

        sentences = sorted(
            (
                serialize_sample(
                    sample
                )
                for sample
                in result[
                    "sentence_samples"
                ][split].values()
            ),
            key=lambda item:
                item["sample_id"],
        )

        contexts = sorted(
            (
                serialize_sample(
                    sample
                )
                for sample
                in result[
                    "context_samples"
                ][split].values()
            ),
            key=lambda item:
                item["sample_id"],
        )

        write_jsonl(
            sentences,
            output_dir
            /
            "sentence_level"
            /
            f"{split}.jsonl",
        )

        write_jsonl(
            contexts,
            output_dir
            /
            "context_level"
            /
            f"{split}.jsonl",
        )

        report[
            "splits"
        ][split] = {
            "sentence_samples":
                len(sentences),

            "context_samples":
                len(contexts),

            "input_context_occurrences":
                result[
                    "stats"
                ][split].get(
                    "input_context_occurrences",
                    0,
                ),

            "rejected_context_occurrences":
                result[
                    "stats"
                ][split].get(
                    "rejected_context_occurrences",
                    0,
                ),
        }

    write_jsonl(
        serialize_rejected(
            result[
                "rejected_sentences"
            ]
        ),
        output_dir
        /
        "rejected"
        /
        "leakage_sentences.jsonl",
    )

    write_jsonl(
        serialize_rejected(
            result[
                "rejected_contexts"
            ]
        ),
        output_dir
        /
        "rejected"
        /
        "leakage_contexts.jsonl",
    )

    write_jsonl(
        result[
            "invalid_contexts"
        ],
        output_dir
        /
        "rejected"
        /
        "invalid_contexts.jsonl",
    )

    write_jsonl(
        ownership[
            "errors"
        ],
        output_dir
        /
        "rejected"
        /
        "ownership_errors.jsonl",
    )

    report_path = (
        output_dir
        /
        "build_report.json"
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
        "FEVER Benign Context Builder"
    )

    print(
        "================================"
    )

    for split in SPLITS:

        split_report = (
            report[
                "splits"
            ][split]
        )

        print(
            f"\n{split}"
        )

        print(
            "  sentence samples:",
            split_report[
                "sentence_samples"
            ],
        )

        print(
            "  context samples:",
            split_report[
                "context_samples"
            ],
        )

        print(
            "  input context occurrences:",
            split_report[
                "input_context_occurrences"
            ],
        )

        print(
            "  rejected context occurrences:",
            split_report[
                "rejected_context_occurrences"
            ],
        )

    print(
        "\nOwnership errors:",
        len(
            ownership[
                "errors"
            ]
        ),
    )

    print(
        "Invalid contexts:",
        len(
            result[
                "invalid_contexts"
            ]
        ),
    )

    print(
        "\nOutput:",
        output_dir,
    )

if __name__ == "__main__":
    main()