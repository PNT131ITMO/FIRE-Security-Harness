from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "paper_dev", "paper_test")
SPLIT_PRIORITY = ("paper_test", "paper_dev", "train")

ATTACK_FIELDS = (
    "attack_mode",
    "attack_family",
    "attack_objective",
    "template_id",
    "template_split",
    "insertion_position",
    "representation",
    "encoding",
)

REJECTED_FILES = (
    "leakage_sentences.jsonl",
    "leakage_contexts.jsonl",
    "invalid_contexts.jsonl",
    "ownership_errors.jsonl",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_jsonl(
    path: Path,
) -> list[dict]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for (
            line_number,
            line,
        ) in enumerate(
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
                    f"{path}:"
                    f"{line_number}: "
                    f"{error}"
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


def get_fever_label(
    record: dict,
):

    return record.get(
        "fever_label",
        record.get(
            "label"
        ),
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

    if not isinstance(
        value,
        list,
    ):

        raise ValueError(
            "Missing evidence sets"
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
                    evidence_set.get(
                        key
                    )
                )

                if isinstance(
                    candidate,
                    list,
                ):

                    items = candidate
                    break

            if items is None:

                raise ValueError(
                    "Unsupported "
                    "evidence-set schema"
                )

        else:

            raise ValueError(
                "Unsupported "
                "evidence-set schema"
            )

        result.append(
            items
        )

    return result


def parse_source_evidence(
    item: dict,
) -> dict:

    page = item[
        "page"
    ]

    sentence_id = item[
        "sentence_id"
    ]

    text = item[
        "text"
    ]

    canonical_page = item.get(
        "canonical_page"
    )

    if not isinstance(
        canonical_page,
        str,
    ):

        canonical_page = page

    canonical_page = (
        normalize_page(
            canonical_page
        )
    )

    return {
        "ref": (
            canonical_page,
            sentence_id,
        ),

        "page":
            page,

        "text":
            text,

        "canonical_page":
            canonical_page,

        "sentence_id":
            sentence_id,
    }


def build_source_index(
    input_dir: Path,
) -> dict:

    result = {}

    for split in SPLITS:

        refs = {}

        ref_pages = defaultdict(
            set
        )

        ref_claims = defaultdict(
            set
        )

        ref_labels = defaultdict(
            set
        )

        contexts = {}

        context_claims = defaultdict(
            set
        )

        context_labels = defaultdict(
            set
        )

        path = (
            input_dir
            /
            f"{split}.jsonl"
        )

        for record in read_jsonl(
            path
        ):

            claim_id = record.get(
                "claim_id",
                record.get(
                    "id"
                ),
            )

            fever_label = (
                get_fever_label(
                    record
                )
            )

            for evidence_set in (
                get_evidence_sets(
                    record
                )
            ):

                parsed = [
                    parse_source_evidence(
                        item
                    )
                    for item
                    in evidence_set
                ]

                for item in parsed:

                    ref = item[
                        "ref"
                    ]

                    refs.setdefault(
                        ref,
                        item[
                            "text"
                        ],
                    )

                    ref_pages[
                        ref
                    ].add(
                        item[
                            "page"
                        ]
                    )

                    if (
                        claim_id
                        is not None
                    ):

                        ref_claims[
                            ref
                        ].add(
                            claim_id
                        )

                    if (
                        fever_label
                        is not None
                    ):

                        ref_labels[
                            ref
                        ].add(
                            str(
                                fever_label
                            )
                        )

                ordered = sorted(
                    parsed,
                    key=lambda item: (
                        item[
                            "canonical_page"
                        ],
                        item[
                            "sentence_id"
                        ],
                    ),
                )

                context_key = tuple(
                    item[
                        "ref"
                    ]
                    for item
                    in ordered
                )

                context_text = (
                    "\n".join(
                        item[
                            "text"
                        ]
                        for item
                        in ordered
                    )
                )

                contexts.setdefault(
                    context_key,
                    context_text,
                )

                if (
                    claim_id
                    is not None
                ):

                    context_claims[
                        context_key
                    ].add(
                        claim_id
                    )

                if (
                    fever_label
                    is not None
                ):

                    context_labels[
                        context_key
                    ].add(
                        str(
                            fever_label
                        )
                    )

        result[
            split
        ] = {

            "refs":
                refs,

            "ref_pages":
                ref_pages,

            "ref_claims":
                ref_claims,

            "ref_labels":
                ref_labels,

            "contexts":
                contexts,

            "context_claims":
                context_claims,

            "context_labels":
                context_labels,
        }

    return result


def record_error(
    errors: list,
    split: str,
    granularity: str,
    sample_id,
    reason: str,
) -> None:

    errors.append(
        {
            "split":
                split,

            "granularity":
                granularity,

            "sample_id":
                sample_id,

            "reason":
                reason,
        }
    )


def validate_common(
    record: dict,
    split: str,
    granularity: str,
    errors: list,
) -> None:

    sample_id = record.get(
        "sample_id"
    )

    if (
        not isinstance(
            sample_id,
            str,
        )
        or not sample_id
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "sample_id_missing",
        )

    if (
        record.get(
            "label"
        )
        != "benign"
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "label_not_benign",
        )

    if (
        record.get(
            "source"
        )
        != "fever"
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "source_not_fever",
        )

    if (
        record.get(
            "original_source"
        )
        != "FEVER"
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "original_source_not_fever",
        )

    if (
        record.get(
            "synthetic"
        )
        is not False
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "synthetic_not_false",
        )

    if (
        record.get(
            "granularity"
        )
        != granularity
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "granularity_mismatch",
        )

    if (
        record.get(
            "upstream_split"
        )
        != split
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "upstream_split_mismatch",
        )

    if (
        record.get(
            "split"
        )
        != split
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "split_mismatch",
        )

    expected_locked = (
        split
        == "paper_test"
    )

    if (
        record.get(
            "locked"
        )
        is not expected_locked
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "locked_flag_mismatch",
        )

    for field in ATTACK_FIELDS:

        if (
            record.get(
                field
            )
            is not None
        ):

            record_error(
                errors,
                split,
                granularity,
                sample_id,
                (
                    f"{field}"
                    "_must_be_null"
                ),
            )

    if (
        "claim_id"
        in record
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "deprecated_claim_id_present",
        )

    claim_ids = record.get(
        "claim_ids"
    )

    if not isinstance(
        claim_ids,
        list,
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "claim_ids_missing",
        )

    else:

        if (
            len(
                claim_ids
            )
            !=
            len(
                set(
                    claim_ids
                )
            )
        ):

            record_error(
                errors,
                split,
                granularity,
                sample_id,
                "duplicate_claim_ids",
            )

        sorted_claim_ids = sorted(
            claim_ids,
            key=str,
        )

        if (
            claim_ids
            != sorted_claim_ids
        ):

            record_error(
                errors,
                split,
                granularity,
                sample_id,
                "claim_ids_not_sorted",
            )

        expected_primary = (
            sorted_claim_ids[0]
            if sorted_claim_ids
            else None
        )

        if (
            record.get(
                "primary_claim_id"
            )
            != expected_primary
        ):

            record_error(
                errors,
                split,
                granularity,
                sample_id,
                "primary_claim_id_mismatch",
            )

    fever_labels = record.get(
        "fever_labels"
    )

    if not isinstance(
        fever_labels,
        list,
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "fever_labels_missing",
        )

    else:

        if (
            len(
                fever_labels
            )
            !=
            len(
                set(
                    fever_labels
                )
            )
        ):

            record_error(
                errors,
                split,
                granularity,
                sample_id,
                "duplicate_fever_labels",
            )

        if (
            fever_labels
            != sorted(
                fever_labels
            )
        ):

            record_error(
                errors,
                split,
                granularity,
                sample_id,
                "fever_labels_not_sorted",
            )

    if not isinstance(
        record.get(
            "evidence_refs"
        ),
        list,
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "evidence_refs_missing",
        )

    base_text_id = record.get(
        "base_text_id"
    )

    if (
        not isinstance(
            base_text_id,
            str,
        )
        or not base_text_id
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "base_text_id_missing",
        )

    text = record.get(
        "text"
    )

    if (
        not isinstance(
            text,
            str,
        )
        or not text.strip()
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "empty_text",
        )

        return

    if (
        record.get(
            "text_sha256"
        )
        != sha256_text(
            text
        )
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "text_sha256_mismatch",
        )

    expected_normalized_hash = (
        sha256_text(
            normalize_text(
                text
            )
        )
    )

    if (
        record.get(
            "normalized_text_sha256"
        )
        !=
        expected_normalized_hash
    ):

        record_error(
            errors,
            split,
            granularity,
            sample_id,
            "normalized_text_sha256_mismatch",
        )


def ref_from_output(
    item: dict,
) -> tuple[str, int]:

    return (
        normalize_page(
            item[
                "canonical_page"
            ]
        ),
        item[
            "sentence_id"
        ],
    )


def validate_sentence(
    record: dict,
    split: str,
    source_index: dict,
    errors: list,
) -> None:

    sample_id = record.get(
        "sample_id"
    )

    refs = record.get(
        "evidence_refs",
        [],
    )

    if (
        len(refs)
        != 1
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "sentence_must_have_one_ref",
        )

        return

    if (
        record.get(
            "context_size"
        )
        != 1
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "context_size_mismatch",
        )

    if (
        record.get(
            "context_complete"
        )
        is not None
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "sentence_context_complete_must_be_null",
        )

    try:

        ref = ref_from_output(
            refs[0]
        )

    except Exception:

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "invalid_evidence_ref",
        )

        return

    if (
        ref
        not in source_index[
            "refs"
        ]
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "ref_not_found_in_gold_source",
        )

        return

    if (
        record.get(
            "text"
        )
        !=
        source_index[
            "refs"
        ][ref]
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "text_not_equal_to_gold_evidence",
        )

    if (
        record.get(
            "canonical_page"
        )
        != ref[0]
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "canonical_page_mismatch",
        )

    page = record.get(
        "page"
    )

    if (
        not isinstance(
            page,
            str,
        )
        or
        normalize_page(
            page
        )
        != ref[0]
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "page_mismatch",
        )

    if (
        record.get(
            "sentence_id"
        )
        != ref[1]
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "sentence_id_mismatch",
        )

    ref_item = refs[0]

    ref_page = ref_item.get(
        "page"
    )

    ref_canonical_page = (
        ref_item.get(
            "canonical_page"
        )
    )

    if (
        not isinstance(
            ref_page,
            str,
        )
        or
        normalize_page(
            ref_page
        )
        != ref[0]
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "evidence_ref_page_mismatch",
        )

    if (
        not isinstance(
            ref_canonical_page,
            str,
        )
        or
        normalize_page(
            ref_canonical_page
        )
        != ref[0]
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "evidence_ref_canonical_page_mismatch",
        )

    if (
        ref_item.get(
            "sentence_id"
        )
        != ref[1]
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "evidence_ref_sentence_id_mismatch",
        )

    expected_identity = (
        f"sentence\u241f"
        f"{ref[0]}"
        f"\u241f"
        f"{ref[1]}"
    )

    expected_hash = (
        sha256_text(
            expected_identity
        )
    )

    expected_base_text_id = (
        f"fever:sentence:"
        f"{expected_hash}"
    )

    expected_sample_id = (
        f"fever_sentence_"
        f"{expected_hash}"
    )

    if (
        record.get(
            "base_text_sha256"
        )
        != expected_hash
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "base_text_sha256_mismatch",
        )

    if (
        record.get(
            "base_text_id"
        )
        !=
        expected_base_text_id
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "base_text_id_mismatch",
        )

    if (
        sample_id
        != expected_sample_id
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "sample_id_mismatch",
        )

    claims = set(
        record.get(
            "claim_ids",
            [],
        )
    )

    source_claims = set(
        source_index[
            "ref_claims"
        ][ref]
    )

    if (
        claims
        != source_claims
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "claim_provenance_mismatch",
        )

    labels = set(
        record.get(
            "fever_labels",
            [],
        )
    )

    source_labels = set(
        source_index[
            "ref_labels"
        ][ref]
    )

    if (
        labels
        != source_labels
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "fever_label_provenance_mismatch",
        )

    original_pages = (
        record.get(
            "original_pages"
        )
    )

    expected_original_pages = (
        sorted(
            source_index[
                "ref_pages"
            ][ref]
        )
    )

    if (
        original_pages
        !=
        expected_original_pages
    ):

        record_error(
            errors,
            split,
            "sentence",
            sample_id,
            "original_pages_mismatch",
        )


def validate_context(
    record: dict,
    split: str,
    source_index: dict,
    errors: list,
) -> None:

    sample_id = record.get(
        "sample_id"
    )

    if (
        record.get(
            "context_complete"
        )
        is not True
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "context_not_complete",
        )

    if (
        record.get(
            "page"
        )
        is not None
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "context_page_must_be_null",
        )

    if (
        record.get(
            "canonical_page"
        )
        is not None
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "context_canonical_page_must_be_null",
        )

    if (
        record.get(
            "sentence_id"
        )
        is not None
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "context_sentence_id_must_be_null",
        )

    refs_raw = record.get(
        "evidence_refs",
        [],
    )

    if not refs_raw:

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "empty_context_refs",
        )

        return

    if (
        record.get(
            "context_size"
        )
        != len(
            refs_raw
        )
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "context_size_mismatch",
        )

    try:

        refs = [
            ref_from_output(
                item
            )
            for item
            in refs_raw
        ]

    except Exception:

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "invalid_context_ref",
        )

        return

    if (
        len(refs)
        !=
        len(
            set(
                refs
            )
        )
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "duplicate_ref_in_context",
        )

    context_key = tuple(
        sorted(
            refs
        )
    )

    if (
        tuple(
            refs
        )
        != context_key
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "context_refs_not_canonical_order",
        )

    if (
        context_key
        not in
        source_index[
            "contexts"
        ]
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "context_not_found_in_gold_source",
        )

        return

    expected_text = (
        source_index[
            "contexts"
        ][
            context_key
        ]
    )

    if (
        record.get(
            "text"
        )
        != expected_text
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "context_text_mismatch",
        )

    for (
        item,
        ref,
    ) in zip(
        refs_raw,
        refs,
    ):

        raw_page = item.get(
            "page"
        )

        canonical_page = item.get(
            "canonical_page"
        )

        if (
            not isinstance(
                raw_page,
                str,
            )
            or
            normalize_page(
                raw_page
            )
            != ref[0]
        ):

            record_error(
                errors,
                split,
                "context",
                sample_id,
                "context_evidence_ref_page_mismatch",
            )

        if (
            not isinstance(
                canonical_page,
                str,
            )
            or
            normalize_page(
                canonical_page
            )
            != ref[0]
        ):

            record_error(
                errors,
                split,
                "context",
                sample_id,
                "context_evidence_ref_canonical_page_mismatch",
            )

        if (
            item.get(
                "sentence_id"
            )
            != ref[1]
        ):

            record_error(
                errors,
                split,
                "context",
                sample_id,
                "context_evidence_ref_sentence_id_mismatch",
            )

    identity_refs = [
        (
            f"{page}"
            f"\u241f"
            f"{sentence_id}"
        )
        for (
            page,
            sentence_id,
        ) in context_key
    ]

    expected_identity = (
        "context\u241f"
        +
        "\u241e".join(
            identity_refs
        )
    )

    expected_hash = (
        sha256_text(
            expected_identity
        )
    )

    expected_base_text_id = (
        f"fever:context:"
        f"{expected_hash}"
    )

    expected_sample_id = (
        f"fever_context_"
        f"{expected_hash}"
    )

    if (
        record.get(
            "base_text_sha256"
        )
        != expected_hash
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "base_text_sha256_mismatch",
        )

    if (
        record.get(
            "base_text_id"
        )
        !=
        expected_base_text_id
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "base_text_id_mismatch",
        )

    if (
        sample_id
        !=
        expected_sample_id
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "sample_id_mismatch",
        )

    claims = set(
        record.get(
            "claim_ids",
            [],
        )
    )

    source_claims = set(
        source_index[
            "context_claims"
        ][
            context_key
        ]
    )

    if (
        claims
        != source_claims
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "claim_provenance_mismatch",
        )

    labels = set(
        record.get(
            "fever_labels",
            [],
        )
    )

    source_labels = set(
        source_index[
            "context_labels"
        ][
            context_key
        ]
    )

    if (
        labels
        != source_labels
    ):

        record_error(
            errors,
            split,
            "context",
            sample_id,
            "fever_label_provenance_mismatch",
        )


def collect_cross_split(
    datasets: dict,
    source_index: dict,
) -> dict:

    pages = defaultdict(
        set
    )

    refs = defaultdict(
        set
    )

    normalized_evidence_texts = (
        defaultdict(
            set
        )
    )

    normalized_sample_texts = (
        defaultdict(
            set
        )
    )

    for split in SPLITS:

        for granularity in (
            "sentence",
            "context",
        ):

            for record in (
                datasets[
                    granularity
                ][split]
            ):

                normalized_hash = (
                    record.get(
                        "normalized_text_sha256"
                    )
                )

                if normalized_hash:

                    normalized_sample_texts[
                        normalized_hash
                    ].add(
                        split
                    )

                for item in (
                    record.get(
                        "evidence_refs",
                        [],
                    )
                ):

                    try:

                        ref = (
                            ref_from_output(
                                item
                            )
                        )

                    except Exception:

                        continue

                    pages[
                        ref[0]
                    ].add(
                        split
                    )

                    refs[
                        ref
                    ].add(
                        split
                    )

                    source_text = (
                        source_index[
                            split
                        ][
                            "refs"
                        ].get(
                            ref
                        )
                    )

                    if (
                        source_text
                        is not None
                    ):

                        text_hash = (
                            sha256_text(
                                normalize_text(
                                    source_text
                                )
                            )
                        )

                        normalized_evidence_texts[
                            text_hash
                        ].add(
                            split
                        )

    def overlap_count(
        store,
    ):

        return sum(
            len(
                value
            ) > 1
            for value
            in store.values()
        )

    return {

        "page_overlap":
            overlap_count(
                pages
            ),

        "evidence_ref_overlap":
            overlap_count(
                refs
            ),

        "normalized_text_overlap":
            overlap_count(
                normalized_evidence_texts
            ),

        "sample_normalized_text_overlap":
            overlap_count(
                normalized_sample_texts
            ),
    }


def collect_granularity_overlap(
    datasets: dict,
) -> dict:

    result = {}

    for split in SPLITS:

        sentence_hashes = {
            record.get(
                "normalized_text_sha256"
            )
            for record
            in datasets[
                "sentence"
            ][split]
            if record.get(
                "normalized_text_sha256"
            )
        }

        context_hashes = {
            record.get(
                "normalized_text_sha256"
            )
            for record
            in datasets[
                "context"
            ][split]
            if record.get(
                "normalized_text_sha256"
            )
        }

        result[
            split
        ] = len(
            sentence_hashes
            &
            context_hashes
        )

    return result


def validate_ids(
    datasets: dict,
    errors: list,
) -> None:

    sample_ids = {}
    base_ids = {}

    for granularity in (
        "sentence",
        "context",
    ):

        for split in SPLITS:

            for record in (
                datasets[
                    granularity
                ][split]
            ):

                sample_id = (
                    record.get(
                        "sample_id"
                    )
                )

                base_id = (
                    record.get(
                        "base_text_id"
                    )
                )

                location = (
                    granularity,
                    split,
                )

                if (
                    sample_id
                    in sample_ids
                ):

                    errors.append(
                        {
                            "reason":
                                "duplicate_sample_id",

                            "sample_id":
                                sample_id,

                            "first":
                                sample_ids[
                                    sample_id
                                ],

                            "second":
                                location,
                        }
                    )

                else:

                    sample_ids[
                        sample_id
                    ] = location

                if (
                    base_id
                    in base_ids
                ):

                    errors.append(
                        {
                            "reason":
                                "duplicate_base_text_id",

                            "base_text_id":
                                base_id,

                            "first":
                                base_ids[
                                    base_id
                                ],

                            "second":
                                location,
                        }
                    )

                else:

                    base_ids[
                        base_id
                    ] = location


def collect_rejection_statistics(
    dataset_dir: Path,
) -> dict:

    rejected_dir = (
        dataset_dir
        /
        "rejected"
    )

    result = {}

    for file_name in (
        REJECTED_FILES
    ):

        path = (
            rejected_dir
            /
            file_name
        )

        if not path.exists():

            result[
                file_name
            ] = {

                "records":
                    0,

                "exists":
                    False,

                "reasons":
                    {},
            }

            continue

        records = read_jsonl(
            path
        )

        reasons = Counter()

        for record in records:

            value = record.get(
                "reasons",
                record.get(
                    "reason"
                ),
            )

            if isinstance(
                value,
                list,
            ):

                reasons.update(
                    str(item)
                    for item
                    in value
                )

            elif (
                value
                is not None
            ):

                reasons[
                    str(
                        value
                    )
                ] += 1

        result[
            file_name
        ] = {

            "records":
                len(
                    records
                ),

            "exists":
                True,

            "reasons":
                dict(
                    sorted(
                        reasons.items()
                    )
                ),
        }

    build_report_path = (
        dataset_dir
        /
        "build_report.json"
    )

    if (
        build_report_path.exists()
    ):

        result[
            "build_report"
        ] = json.loads(
            build_report_path.read_text(
                encoding="utf-8"
            )
        )

    return result


def file_entry(
    path: Path,
    records: int | None = None,
) -> dict:

    entry = {

        "path":
            str(
                path
            ),

        "sha256":
            sha256_file(
                path
            ),
    }

    if (
        records
        is not None
    ):

        entry[
            "records"
        ] = records

    return entry


def main() -> None:

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--dataset-dir",
        default=(
            "data/processed/"
            "defense/sources/fever"
        ),
    )

    parser.add_argument(
        "--gold-dir",
        default=(
            "data/processed/"
            "fever/gold/clean"
        ),
    )

    args = parser.parse_args()

    dataset_dir = Path(
        args.dataset_dir
    )

    gold_dir = Path(
        args.gold_dir
    )

    datasets = {

        "sentence":
            {},

        "context":
            {},
    }

    output_files = []

    for (
        granularity,
        directory,
    ) in (

        (
            "sentence",
            "sentence_level",
        ),

        (
            "context",
            "context_level",
        ),
    ):

        for split in SPLITS:

            path = (
                dataset_dir
                /
                directory
                /
                f"{split}.jsonl"
            )

            datasets[
                granularity
            ][
                split
            ] = read_jsonl(
                path
            )

            output_files.append(
                path
            )

    print(
        "Building Gold Evidence "
        "provenance index..."
    )

    source_index = (
        build_source_index(
            gold_dir
        )
    )

    errors = []
    stats = {}

    for split in SPLITS:

        stats[
            split
        ] = {}

        for granularity in (
            "sentence",
            "context",
        ):

            records = (
                datasets[
                    granularity
                ][
                    split
                ]
            )

            stats[
                split
            ][
                granularity
            ] = len(
                records
            )

            for record in records:

                validate_common(
                    record,
                    split,
                    granularity,
                    errors,
                )

                if (
                    granularity
                    == "sentence"
                ):

                    validate_sentence(
                        record,
                        split,
                        source_index[
                            split
                        ],
                        errors,
                    )

                else:

                    validate_context(
                        record,
                        split,
                        source_index[
                            split
                        ],
                        errors,
                    )

    validate_ids(
        datasets,
        errors,
    )

    leakage = (
        collect_cross_split(
            datasets,
            source_index,
        )
    )

    for name in (

        "page_overlap",

        "evidence_ref_overlap",

        "normalized_text_overlap",
    ):

        if (
            leakage[
                name
            ]
            != 0
        ):

            errors.append(
                {
                    "reason":
                        name,

                    "count":
                        leakage[
                            name
                        ],
                }
            )

    granularity_overlap = (
        collect_granularity_overlap(
            datasets
        )
    )

    validator_path = (
        Path(
            __file__
        ).resolve()
    )

    builder_path = (
        validator_path.parent
        /
        "build_fever_benign_contexts.py"
    )

    if (
        not builder_path.exists()
    ):

        errors.append(
            {
                "reason":
                    "builder_script_missing",

                "path":
                    str(
                        builder_path
                    ),
            }
        )

    status = (
        "PASS"
        if not errors
        else "FAIL"
    )

    error_summary = Counter(
        error.get(
            "reason",
            "unknown",
        )
        for error
        in errors
    )

    paper_test_locked = all(

        record.get(
            "locked"
        )
        is True

        for granularity
        in (
            "sentence",
            "context",
        )

        for record
        in datasets[
            granularity
        ][
            "paper_test"
        ]
    )

    validation_report = {

        "status":
            status,

        "errors":
            len(
                errors
            ),

        "warnings":
            0,

        "split_counts":
            stats,

        "cross_split_leakage":
            leakage,

        "granularity_overlap":
            granularity_overlap,

        "paper_test_locked":
            paper_test_locked,

        "error_summary":
            dict(
                sorted(
                    error_summary.items()
                )
            ),

        "error_details":
            errors[:1000],

        "error_log":
            "validation_errors.jsonl",
    }

    leakage_report = {

        "policy": {

            "split_priority":
                list(
                    SPLIT_PRIORITY
                ),

            "page_disjoint":
                True,

            "evidence_ref_disjoint":
                True,

            "normalized_text_disjoint":
                True,

            "page_normalization":
                "Unicode NFC",

            "audit_text_normalization":
                (
                    "Unicode NFKC + "
                    "strip + collapse "
                    "whitespace + casefold"
                ),

            "model_text":
                "original text preserved",

            "paper_test_locked":
                True,
        },

        "final_cross_split_overlap":
            leakage,

        "within_split_granularity_overlap":
            granularity_overlap,
    }

    rejection_statistics = (
        collect_rejection_statistics(
            dataset_dir
        )
    )

    gold_inputs = {}

    for split in SPLITS:

        path = (
            gold_dir
            /
            f"{split}.jsonl"
        )

        gold_inputs[
            split
        ] = file_entry(
            path,
            len(
                read_jsonl(
                    path
                )
            ),
        )

    pipeline_scripts = {

        "validator":
            file_entry(
                validator_path
            ),

        "builder":
            (
                file_entry(
                    builder_path
                )
                if
                builder_path.exists()
                else
                {
                    "path":
                        str(
                            builder_path
                        ),

                    "exists":
                        False,
                }
            ),
    }

    manifest = {

        "dataset":
            "FEVER Benign "
            "Base Contexts",

        "dataset_version":
            "fever-benign-v1",

        "step":
            "3B",

        "source":
            "FEVER Gold "
            "Evidence Clean",

        "defense_label":
            "benign",

        "synthetic":
            False,

        "primary_granularity":
            "context",

        "sentence_level_role":
            "ablation",

        "combined_training_policy":
            (
                "deduplicate_by_"
                "normalized_text_sha256_"
                "prefer_context"
            ),

        "normalization_policy": {

            "page":
                "Unicode NFC",

            "audit_text":
                (
                    "Unicode NFKC + "
                    "strip + collapse "
                    "whitespace + casefold"
                ),

            "model_text":
                "original text preserved",
        },

        "split_priority":
            list(
                SPLIT_PRIORITY
            ),

        "paper_test_policy": {

            "locked":
                True,

            "allow_training":
                False,

            "allow_model_selection":
                False,

            "allow_threshold_tuning":
                False,

            "allow_preprocessing_selection":
                False,

            "purpose":
                "final_in_domain_evaluation",
        },

        "splits":
            stats,

        "granularity_overlap":
            granularity_overlap,

        "rejection_statistics":
            rejection_statistics,

        "gold_inputs":
            gold_inputs,

        "pipeline_scripts":
            pipeline_scripts,

        "files": {

            str(
                path.relative_to(
                    dataset_dir
                )
            ): {

                "sha256":
                    sha256_file(
                        path
                    ),

                "records":
                    len(
                        read_jsonl(
                            path
                        )
                    ),
            }

            for path
            in output_files
        },
    }

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        errors,
        dataset_dir
        /
        "validation_errors.jsonl",
    )

    (
        dataset_dir
        /
        "validation_report.json"
    ).write_text(
        json.dumps(
            validation_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        dataset_dir
        /
        "leakage_report.json"
    ).write_text(
        json.dumps(
            leakage_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        dataset_dir
        /
        "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n================================"
    )

    print(
        "FEVER Benign Context Validation"
    )

    print(
        "================================"
    )

    for split in SPLITS:

        print(
            f"\n{split}"
        )

        print(
            "  sentence:",
            stats[
                split
            ][
                "sentence"
            ],
        )

        print(
            "  context:",
            stats[
                split
            ][
                "context"
            ],
        )

        print(
            (
                "  sentence/context "
                "text overlap:"
            ),
            granularity_overlap[
                split
            ],
        )

    print(
        "\nCross-split leakage:"
    )

    print(
        "  page:",
        leakage[
            "page_overlap"
        ],
    )

    print(
        "  evidence ref:",
        leakage[
            "evidence_ref_overlap"
        ],
    )

    print(
        (
            "  normalized "
            "evidence text:"
        ),
        leakage[
            "normalized_text_overlap"
        ],
    )

    print(
        (
            "  normalized "
            "sample text:"
        ),
        leakage[
            "sample_normalized_text_overlap"
        ],
    )

    print(
        "\nPaper test locked:",
        paper_test_locked,
    )

    print(
        "Errors:",
        len(
            errors
        ),
    )

    if error_summary:

        print(
            "Error summary:"
        )

        for (
            reason,
            count,
        ) in sorted(
            error_summary.items()
        ):

            print(
                f"  {reason}: "
                f"{count}"
            )

    print(
        "Status:",
        status,
    )

    if errors:

        raise SystemExit(
            1
        )

if __name__ == "__main__":
    main()