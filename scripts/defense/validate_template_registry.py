from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


TEMPLATE_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "ood_template": "ood_template.jsonl",
    "ood_family": "ood_family.jsonl",
}

EXPECTED_TEMPLATE_COUNTS = {
    "train": 36,
    "validation": 12,
    "ood_template": 12,
    "ood_family": 16,
}

EXPECTED_CLUSTER_COUNTS = {
    "train": 18,
    "validation": 6,
    "ood_template": 6,
    "ood_family": 8,
}

EXPECTED_TEMPLATES_PER_CLUSTER = 2
EXPECTED_PAYLOAD_COUNT = 16

ALLOWED_POSITIONS = {
    "prefix",
    "middle",
    "suffix",
}

ALLOWED_REPRESENTATIONS = {
    "plain_text",
    "markdown",
}

ALLOWED_ENCODINGS = {
    "plain",
}

ALLOWED_LANGUAGES = {
    "en",
}

PLACEHOLDER_PATTERN = re.compile(
    r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}"
)


def normalize_text(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    return " ".join(
        text.strip().split()
    ).casefold()


def sha256_text(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def read_json(
    path: Path,
) -> dict:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_jsonl(
    path: Path,
) -> list[dict]:

    result = []

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

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"{path}:{line_number}: {error}"
                ) from error

            result.append(
                record
            )

    return result


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


def add_error(
    errors: list,
    reason: str,
    item_id=None,
    file_name=None,
    details=None,
) -> None:

    error = {
        "reason": reason,
    }

    if item_id is not None:
        error[
            "item_id"
        ] = item_id

    if file_name is not None:
        error[
            "file"
        ] = file_name

    if details is not None:
        error[
            "details"
        ] = details

    errors.append(
        error
    )


def extract_placeholders(
    template: str,
) -> list[str]:

    return PLACEHOLDER_PATTERN.findall(
        template
    )


def validate_family_registry(
    family_registry: dict,
    errors: list,
) -> dict:

    objective_vocabulary = (
        family_registry.get(
            "objective_vocabulary"
        )
    )

    if not isinstance(
        objective_vocabulary,
        list,
    ):

        add_error(
            errors,
            "objective_vocabulary_missing",
            file_name="family_registry.json",
        )

        objective_vocabulary = []

    if (
        len(
            objective_vocabulary
        )
        != len(
            set(
                objective_vocabulary
            )
        )
    ):

        add_error(
            errors,
            "duplicate_objective_vocabulary",
            file_name="family_registry.json",
        )

    objective_set = set(
        objective_vocabulary
    )

    development_families = (
        family_registry.get(
            "development_families"
        )
    )

    ood_families = (
        family_registry.get(
            "ood_families"
        )
    )

    if not isinstance(
        development_families,
        list,
    ):

        add_error(
            errors,
            "development_families_missing",
            file_name="family_registry.json",
        )

        development_families = []

    if not isinstance(
        ood_families,
        list,
    ):

        add_error(
            errors,
            "ood_families_missing",
            file_name="family_registry.json",
        )

        ood_families = []

    development_set = set(
        development_families
    )

    ood_set = set(
        ood_families
    )

    overlap = (
        development_set
        &
        ood_set
    )

    if overlap:

        add_error(
            errors,
            "development_ood_family_overlap",
            file_name="family_registry.json",
            details=sorted(
                overlap
            ),
        )

    classification_policy = (
        family_registry.get(
            "classification_policy"
        )
    )

    expected_classification_policy = {
        "single_primary_family_required":
            True,

        "single_primary_objective_required":
            True,

        "family_assignment_policy":
            "assign_by_attack_mechanism",

        "multi_mechanism_policy":
            "reject",

        "ambiguous_family_policy":
            "reject",
    }

    if (
        classification_policy
        !=
        expected_classification_policy
    ):

        add_error(
            errors,
            "classification_policy_mismatch",
            file_name="family_registry.json",
        )

    semantic_policy = (
        family_registry.get(
            "semantic_cluster_policy"
        )
    )

    if not isinstance(
        semantic_policy,
        dict,
    ):

        add_error(
            errors,
            "semantic_cluster_policy_missing",
            file_name="family_registry.json",
        )

    ood_policy = (
        family_registry.get(
            "ood_family_policy"
        )
    )

    if not isinstance(
        ood_policy,
        dict,
    ):

        add_error(
            errors,
            "ood_family_policy_missing",
            file_name="family_registry.json",
        )

    families_raw = (
        family_registry.get(
            "families"
        )
    )

    if not isinstance(
        families_raw,
        list,
    ):

        add_error(
            errors,
            "families_missing",
            file_name="family_registry.json",
        )

        families_raw = []

    families = {}

    for record in families_raw:

        family = record.get(
            "family"
        )

        if (
            not isinstance(
                family,
                str,
            )
            or not family
        ):

            add_error(
                errors,
                "invalid_family_name",
                file_name="family_registry.json",
            )

            continue

        if family in families:

            add_error(
                errors,
                "duplicate_family",
                item_id=family,
                file_name="family_registry.json",
            )

            continue

        families[
            family
        ] = record

        allowed_objectives = (
            record.get(
                "allowed_objectives"
            )
        )

        if (
            not isinstance(
                allowed_objectives,
                list,
            )
            or not allowed_objectives
        ):

            add_error(
                errors,
                "allowed_objectives_missing",
                item_id=family,
                file_name="family_registry.json",
            )

            continue

        unknown_objectives = (
            set(
                allowed_objectives
            )
            -
            objective_set
        )

        if unknown_objectives:

            add_error(
                errors,
                "unknown_allowed_objective",
                item_id=family,
                file_name="family_registry.json",
                details=sorted(
                    unknown_objectives
                ),
            )

        if not isinstance(
            record.get(
                "definition"
            ),
            str,
        ):

            add_error(
                errors,
                "family_definition_missing",
                item_id=family,
                file_name="family_registry.json",
            )

        if not isinstance(
            record.get(
                "in_scope"
            ),
            list,
        ):

            add_error(
                errors,
                "family_in_scope_missing",
                item_id=family,
                file_name="family_registry.json",
            )

        if not isinstance(
            record.get(
                "out_of_scope"
            ),
            list,
        ):

            add_error(
                errors,
                "family_out_of_scope_missing",
                item_id=family,
                file_name="family_registry.json",
            )

        if (
            record.get(
                "enabled"
            )
            is not True
        ):

            add_error(
                errors,
                "family_not_enabled",
                item_id=family,
                file_name="family_registry.json",
            )

        if family in development_set:

            if (
                record.get(
                    "development_allowed"
                )
                is not True
            ):

                add_error(
                    errors,
                    "development_family_not_allowed",
                    item_id=family,
                    file_name="family_registry.json",
                )

            if (
                record.get(
                    "ood_family"
                )
                is not False
            ):

                add_error(
                    errors,
                    "development_family_marked_ood",
                    item_id=family,
                    file_name="family_registry.json",
                )

        elif family in ood_set:

            if (
                record.get(
                    "development_allowed"
                )
                is not False
            ):

                add_error(
                    errors,
                    "ood_family_development_allowed",
                    item_id=family,
                    file_name="family_registry.json",
                )

            if (
                record.get(
                    "ood_family"
                )
                is not True
            ):

                add_error(
                    errors,
                    "ood_family_flag_missing",
                    item_id=family,
                    file_name="family_registry.json",
                )

        else:

            add_error(
                errors,
                "family_not_assigned_to_split_policy",
                item_id=family,
                file_name="family_registry.json",
            )

    registered_families = set(
        families
    )

    expected_families = (
        development_set
        |
        ood_set
    )

    if (
        registered_families
        != expected_families
    ):

        add_error(
            errors,
            "family_assignment_registry_mismatch",
            file_name="family_registry.json",
            details={
                "missing":
                    sorted(
                        expected_families
                        -
                        registered_families
                    ),

                "extra":
                    sorted(
                        registered_families
                        -
                        expected_families
                    ),
            },
        )

    return {
        "objectives":
            objective_set,

        "development_families":
            development_set,

        "ood_families":
            ood_set,

        "families":
            families,
    }


def validate_payload_registry(
    payloads: list[dict],
    taxonomy: dict,
    errors: list,
) -> dict:

    payload_ids = set()
    normalized_values = {}
    payload_by_key = defaultdict(
        list
    )

    for record in payloads:

        payload_id = record.get(
            "payload_id"
        )

        if (
            not isinstance(
                payload_id,
                str,
            )
            or not payload_id
        ):

            add_error(
                errors,
                "invalid_payload_id",
                file_name="payload_registry.jsonl",
            )

            continue

        if payload_id in payload_ids:

            add_error(
                errors,
                "duplicate_payload_id",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
            )

        payload_ids.add(
            payload_id
        )

        placeholder = record.get(
            "placeholder"
        )

        value = record.get(
            "value"
        )

        objectives = record.get(
            "compatible_objectives"
        )

        if (
            not isinstance(
                placeholder,
                str,
            )
            or not placeholder
        ):

            add_error(
                errors,
                "invalid_payload_placeholder",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
            )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):

            add_error(
                errors,
                "invalid_payload_value",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
            )

            continue

        if (
            not isinstance(
                objectives,
                list,
            )
            or not objectives
        ):

            add_error(
                errors,
                "payload_objectives_missing",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
            )

            objectives = []

        unknown_objectives = (
            set(
                objectives
            )
            -
            taxonomy[
                "objectives"
            ]
        )

        if unknown_objectives:

            add_error(
                errors,
                "payload_unknown_objective",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
                details=sorted(
                    unknown_objectives
                ),
            )

        expected_hash = (
            sha256_text(
                value
            )
        )

        if (
            record.get(
                "payload_sha256"
            )
            != expected_hash
        ):

            add_error(
                errors,
                "payload_sha256_mismatch",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
            )

        normalized = (
            normalize_text(
                value
            )
        )

        existing = (
            normalized_values.get(
                normalized
            )
        )

        if existing is not None:

            add_error(
                errors,
                "duplicate_normalized_payload",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
                details={
                    "first":
                        existing,
                },
            )

        else:

            normalized_values[
                normalized
            ] = payload_id

        if (
            record.get(
                "language"
            )
            not in ALLOWED_LANGUAGES
        ):

            add_error(
                errors,
                "invalid_payload_language",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
            )

        if (
            record.get(
                "source"
            )
            != "handcrafted"
        ):

            add_error(
                errors,
                "invalid_payload_source",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
            )

        if (
            record.get(
                "version"
            )
            != "1.0"
        ):

            add_error(
                errors,
                "invalid_payload_version",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
            )

        if (
            record.get(
                "enabled"
            )
            is not True
        ):

            add_error(
                errors,
                "payload_not_enabled",
                item_id=payload_id,
                file_name="payload_registry.jsonl",
            )

        for objective in objectives:

            payload_by_key[
                (
                    placeholder,
                    objective,
                )
            ].append(
                payload_id
            )

    return {
        "payload_ids":
            payload_ids,

        "payload_by_key":
            payload_by_key,
    }


def validate_template_record(
    record: dict,
    split: str,
    file_name: str,
    taxonomy: dict,
    payload_index: dict,
    errors: list,
) -> None:

    template_id = record.get(
        "template_id"
    )

    family = record.get(
        "family"
    )

    objective = record.get(
        "attack_objective"
    )

    cluster = record.get(
        "semantic_cluster_id"
    )

    template = record.get(
        "template"
    )

    if (
        not isinstance(
            template_id,
            str,
        )
        or not template_id
    ):

        add_error(
            errors,
            "invalid_template_id",
            file_name=file_name,
        )

        return

    expected_prefix = (
        f"{family}_{split}_"
    )

    if not template_id.startswith(
        expected_prefix
    ):

        add_error(
            errors,
            "template_id_prefix_mismatch",
            item_id=template_id,
            file_name=file_name,
            details={
                "expected_prefix":
                    expected_prefix,
            },
        )

    suffix = template_id[
        len(
            expected_prefix
        ):
    ]

    if (
        len(
            suffix
        )
        != 3
        or not suffix.isdigit()
    ):

        add_error(
            errors,
            "template_id_suffix_invalid",
            item_id=template_id,
            file_name=file_name,
        )

    if (
        record.get(
            "template_split"
        )
        != split
    ):

        add_error(
            errors,
            "template_split_mismatch",
            item_id=template_id,
            file_name=file_name,
        )

    family_record = (
        taxonomy[
            "families"
        ].get(
            family
        )
    )

    if family_record is None:

        add_error(
            errors,
            "unknown_template_family",
            item_id=template_id,
            file_name=file_name,
            details=family,
        )

    else:

        if (
            objective
            not in family_record[
                "allowed_objectives"
            ]
        ):

            add_error(
                errors,
                "objective_not_allowed_for_family",
                item_id=template_id,
                file_name=file_name,
                details={
                    "family":
                        family,

                    "objective":
                        objective,
                },
            )

    if (
        objective
        not in taxonomy[
            "objectives"
        ]
    ):

        add_error(
            errors,
            "unknown_template_objective",
            item_id=template_id,
            file_name=file_name,
        )

    if split in {
        "train",
        "validation",
        "ood_template",
    }:

        if (
            family
            not in taxonomy[
                "development_families"
            ]
        ):

            add_error(
                errors,
                "non_development_family_in_development_split",
                item_id=template_id,
                file_name=file_name,
            )

    elif split == "ood_family":

        if (
            family
            not in taxonomy[
                "ood_families"
            ]
        ):

            add_error(
                errors,
                "non_ood_family_in_ood_family_split",
                item_id=template_id,
                file_name=file_name,
            )

    if (
        not isinstance(
            cluster,
            str,
        )
        or not cluster
    ):

        add_error(
            errors,
            "semantic_cluster_missing",
            item_id=template_id,
            file_name=file_name,
        )

    if (
        not isinstance(
            template,
            str,
        )
        or not template.strip()
    ):

        add_error(
            errors,
            "template_text_missing",
            item_id=template_id,
            file_name=file_name,
        )

        return

    expected_hash = (
        sha256_text(
            template
        )
    )

    if (
        record.get(
            "template_sha256"
        )
        != expected_hash
    ):

        add_error(
            errors,
            "template_sha256_mismatch",
            item_id=template_id,
            file_name=file_name,
        )

    placeholders = record.get(
        "placeholders"
    )

    if not isinstance(
        placeholders,
        list,
    ):

        add_error(
            errors,
            "template_placeholders_missing",
            item_id=template_id,
            file_name=file_name,
        )

        placeholders = []

    extracted = (
        extract_placeholders(
            template
        )
    )

    if (
        len(
            extracted
        )
        != len(
            set(
                extracted
            )
        )
    ):

        add_error(
            errors,
            "duplicate_placeholder_in_template",
            item_id=template_id,
            file_name=file_name,
        )

    if (
        set(
            placeholders
        )
        != set(
            extracted
        )
    ):

        add_error(
            errors,
            "placeholder_declaration_mismatch",
            item_id=template_id,
            file_name=file_name,
            details={
                "declared":
                    placeholders,

                "extracted":
                    extracted,
            },
        )

    if (
        len(
            placeholders
        )
        != len(
            set(
                placeholders
            )
        )
    ):

        add_error(
            errors,
            "duplicate_declared_placeholder",
            item_id=template_id,
            file_name=file_name,
        )

    for placeholder in placeholders:

        key = (
            placeholder,
            objective,
        )

        if (
            key
            not in payload_index[
                "payload_by_key"
            ]
        ):

            add_error(
                errors,
                "no_compatible_payload",
                item_id=template_id,
                file_name=file_name,
                details={
                    "placeholder":
                        placeholder,

                    "objective":
                        objective,
                },
            )

    positions = record.get(
        "allowed_positions"
    )

    if (
        not isinstance(
            positions,
            list,
        )
        or not positions
    ):

        add_error(
            errors,
            "allowed_positions_missing",
            item_id=template_id,
            file_name=file_name,
        )

    else:

        unknown = (
            set(
                positions
            )
            -
            ALLOWED_POSITIONS
        )

        if unknown:

            add_error(
                errors,
                "unknown_allowed_position",
                item_id=template_id,
                file_name=file_name,
                details=sorted(
                    unknown
                ),
            )

        if (
            len(
                positions
            )
            != len(
                set(
                    positions
                )
            )
        ):

            add_error(
                errors,
                "duplicate_allowed_position",
                item_id=template_id,
                file_name=file_name,
            )

    representations = record.get(
        "allowed_representations"
    )

    if (
        not isinstance(
            representations,
            list,
        )
        or not representations
    ):

        add_error(
            errors,
            "allowed_representations_missing",
            item_id=template_id,
            file_name=file_name,
        )

    else:

        unknown = (
            set(
                representations
            )
            -
            ALLOWED_REPRESENTATIONS
        )

        if unknown:

            add_error(
                errors,
                "unknown_allowed_representation",
                item_id=template_id,
                file_name=file_name,
                details=sorted(
                    unknown
                ),
            )

    encodings = record.get(
        "allowed_encodings"
    )

    if (
        not isinstance(
            encodings,
            list,
        )
        or not encodings
    ):

        add_error(
            errors,
            "allowed_encodings_missing",
            item_id=template_id,
            file_name=file_name,
        )

    else:

        unknown = (
            set(
                encodings
            )
            -
            ALLOWED_ENCODINGS
        )

        if unknown:

            add_error(
                errors,
                "unknown_allowed_encoding",
                item_id=template_id,
                file_name=file_name,
                details=sorted(
                    unknown
                ),
            )

    if (
        record.get(
            "language"
        )
        not in ALLOWED_LANGUAGES
    ):

        add_error(
            errors,
            "invalid_template_language",
            item_id=template_id,
            file_name=file_name,
        )

    if (
        record.get(
            "source"
        )
        != "handcrafted"
    ):

        add_error(
            errors,
            "invalid_template_source",
            item_id=template_id,
            file_name=file_name,
        )

    if (
        record.get(
            "parent_template_id"
        )
        is not None
    ):

        add_error(
            errors,
            "parent_template_id_must_be_null_v1",
            item_id=template_id,
            file_name=file_name,
        )

    if (
        record.get(
            "version"
        )
        != "1.0"
    ):

        add_error(
            errors,
            "invalid_template_version",
            item_id=template_id,
            file_name=file_name,
        )

    if (
        record.get(
            "enabled"
        )
        is not True
    ):

        add_error(
            errors,
            "template_not_enabled",
            item_id=template_id,
            file_name=file_name,
        )


def validate_template_registry(
    templates: dict[str, list[dict]],
    taxonomy: dict,
    payload_index: dict,
    errors: list,
) -> dict:

    template_ids = {}
    raw_templates = {}
    normalized_templates = {}

    clusters = defaultdict(
        list
    )

    cluster_splits = defaultdict(
        set
    )

    cluster_families = defaultdict(
        set
    )

    family_split_counts = defaultdict(
        Counter
    )

    objective_by_family_split = defaultdict(
        set
    )

    for split, records in (
        templates.items()
    ):

        file_name = (
            TEMPLATE_FILES[
                split
            ]
        )

        expected_count = (
            EXPECTED_TEMPLATE_COUNTS[
                split
            ]
        )

        if (
            len(
                records
            )
            != expected_count
        ):

            add_error(
                errors,
                "template_count_mismatch",
                file_name=file_name,
                details={
                    "expected":
                        expected_count,

                    "actual":
                        len(
                            records
                        ),
                },
            )

        for record in records:

            validate_template_record(
                record,
                split,
                file_name,
                taxonomy,
                payload_index,
                errors,
            )

            template_id = (
                record.get(
                    "template_id"
                )
            )

            family = record.get(
                "family"
            )

            cluster = record.get(
                "semantic_cluster_id"
            )

            objective = record.get(
                "attack_objective"
            )

            template = record.get(
                "template"
            )

            if template_id is not None:

                if template_id in template_ids:

                    add_error(
                        errors,
                        "duplicate_template_id",
                        item_id=template_id,
                        file_name=file_name,
                        details={
                            "first_split":
                                template_ids[
                                    template_id
                                ],

                            "second_split":
                                split,
                        },
                    )

                else:

                    template_ids[
                        template_id
                    ] = split

            if isinstance(
                template,
                str,
            ):

                existing_raw = (
                    raw_templates.get(
                        template
                    )
                )

                if existing_raw is not None:

                    add_error(
                        errors,
                        "duplicate_raw_template",
                        item_id=template_id,
                        file_name=file_name,
                        details={
                            "first":
                                existing_raw,
                        },
                    )

                else:

                    raw_templates[
                        template
                    ] = template_id

                normalized = (
                    normalize_text(
                        template
                    )
                )

                existing_normalized = (
                    normalized_templates.get(
                        normalized
                    )
                )

                if (
                    existing_normalized
                    is not None
                ):

                    add_error(
                        errors,
                        "duplicate_normalized_template",
                        item_id=template_id,
                        file_name=file_name,
                        details={
                            "first":
                                existing_normalized,
                        },
                    )

                else:

                    normalized_templates[
                        normalized
                    ] = template_id

            if (
                isinstance(
                    cluster,
                    str,
                )
                and cluster
            ):

                clusters[
                    cluster
                ].append(
                    template_id
                )

                cluster_splits[
                    cluster
                ].add(
                    split
                )

                if family is not None:

                    cluster_families[
                        cluster
                    ].add(
                        family
                    )

            if family is not None:

                family_split_counts[
                    split
                ][
                    family
                ] += 1

                if objective is not None:

                    objective_by_family_split[
                        (
                            split,
                            family,
                        )
                    ].add(
                        objective
                    )

    for cluster, split_set in (
        cluster_splits.items()
    ):

        if (
            len(
                split_set
            )
            != 1
        ):

            add_error(
                errors,
                "semantic_cluster_cross_split",
                item_id=cluster,
                details=sorted(
                    split_set
                ),
            )

    for cluster, families in (
        cluster_families.items()
    ):

        if (
            len(
                families
            )
            != 1
        ):

            add_error(
                errors,
                "semantic_cluster_cross_family",
                item_id=cluster,
                details=sorted(
                    families
                ),
            )

    for cluster, template_list in (
        clusters.items()
    ):

        if (
            len(
                template_list
            )
            !=
            EXPECTED_TEMPLATES_PER_CLUSTER
        ):

            add_error(
                errors,
                "templates_per_cluster_mismatch",
                item_id=cluster,
                details={
                    "expected":
                        EXPECTED_TEMPLATES_PER_CLUSTER,

                    "actual":
                        len(
                            template_list
                        ),
                },
            )

    for split in TEMPLATE_FILES:

        split_clusters = {
            cluster
            for cluster, split_set
            in cluster_splits.items()
            if split in split_set
        }

        expected = (
            EXPECTED_CLUSTER_COUNTS[
                split
            ]
        )

        if (
            len(
                split_clusters
            )
            != expected
        ):

            add_error(
                errors,
                "semantic_cluster_count_mismatch",
                file_name=TEMPLATE_FILES[
                    split
                ],
                details={
                    "expected":
                        expected,

                    "actual":
                        len(
                            split_clusters
                        ),
                },
            )

    development_families = (
        taxonomy[
            "development_families"
        ]
    )

    ood_families = (
        taxonomy[
            "ood_families"
        ]
    )

    for family in (
        development_families
    ):

        expected_counts = {
            "train": 6,
            "validation": 2,
            "ood_template": 2,
        }

        for split, expected in (
            expected_counts.items()
        ):

            actual = (
                family_split_counts[
                    split
                ][
                    family
                ]
            )

            if actual != expected:

                add_error(
                    errors,
                    "development_family_template_count_mismatch",
                    item_id=family,
                    file_name=TEMPLATE_FILES[
                        split
                    ],
                    details={
                        "expected":
                            expected,

                        "actual":
                            actual,
                    },
                )

    for family in ood_families:

        actual = (
            family_split_counts[
                "ood_family"
            ][
                family
            ]
        )

        if actual != 8:

            add_error(
                errors,
                "ood_family_template_count_mismatch",
                item_id=family,
                file_name="ood_family.jsonl",
                details={
                    "expected": 8,
                    "actual": actual,
                },
            )

        for split in (
            "train",
            "validation",
            "ood_template",
        ):

            if (
                family_split_counts[
                    split
                ][
                    family
                ]
                != 0
            ):

                add_error(
                    errors,
                    "ood_family_present_in_development_templates",
                    item_id=family,
                    file_name=TEMPLATE_FILES[
                        split
                    ],
                )

    for family in development_families:

        train_objectives = (
            objective_by_family_split[
                (
                    "train",
                    family,
                )
            ]
        )

        validation_objectives = (
            objective_by_family_split[
                (
                    "validation",
                    family,
                )
            ]
        )

        ood_template_objectives = (
            objective_by_family_split[
                (
                    "ood_template",
                    family,
                )
            ]
        )

        unknown_validation = (
            validation_objectives
            -
            train_objectives
        )

        if unknown_validation:

            add_error(
                errors,
                "validation_contains_unseen_objective",
                item_id=family,
                file_name="validation.jsonl",
                details=sorted(
                    unknown_validation
                ),
            )

        known_development_objectives = (
            train_objectives
            |
            validation_objectives
        )

        unknown_ood = (
            ood_template_objectives
            -
            known_development_objectives
        )

        if unknown_ood:

            add_error(
                errors,
                "ood_template_contains_unseen_objective",
                item_id=family,
                file_name="ood_template.jsonl",
                details=sorted(
                    unknown_ood
                ),
            )

    return {
        "template_ids":
            template_ids,

        "clusters":
            clusters,

        "cluster_splits":
            cluster_splits,

        "family_split_counts":
            family_split_counts,

        "objective_by_family_split":
            objective_by_family_split,
    }


def validate_contamination_report(
    registry_dir: Path,
    errors: list,
) -> dict:

    path = (
        registry_dir
        /
        "contamination_report.json"
    )

    if not path.exists():

        add_error(
            errors,
            "contamination_report_missing",
            file_name="contamination_report.json",
        )

        return {}

    report = read_json(
        path
    )

    if (
        report.get(
            "status"
        )
        != "PASS"
    ):

        add_error(
            errors,
            "contamination_status_not_pass",
            file_name="contamination_report.json",
        )

    if (
        report.get(
            "blocking_matches"
        )
        != 0
    ):

        add_error(
            errors,
            "contamination_blocking_matches",
            file_name="contamination_report.json",
            details=report.get(
                "blocking_matches"
            ),
        )

    semantic_check = report.get(
        "semantic_check"
    )

    if (
        not isinstance(
            semantic_check,
            dict,
        )
        or
        semantic_check.get(
            "ran"
        )
        is not True
    ):

        add_error(
            errors,
            "semantic_contamination_check_not_run",
            file_name="contamination_report.json",
        )

    if (
        isinstance(
            semantic_check,
            dict,
        )
        and
        not semantic_check.get(
            "model"
        )
    ):

        add_error(
            errors,
            "semantic_model_missing",
            file_name="contamination_report.json",
        )

    pairs_path = (
        registry_dir
        /
        "contamination_pairs.jsonl"
    )

    if not pairs_path.exists():

        add_error(
            errors,
            "contamination_pairs_missing",
            file_name="contamination_pairs.jsonl",
        )

    return report


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--registry-dir",
        default=(
            "data/processed/defense/"
            "template_registry"
        ),
    )

    args = parser.parse_args()

    registry_dir = Path(
        args.registry_dir
    )

    family_path = (
        registry_dir
        /
        "family_registry.json"
    )

    payload_path = (
        registry_dir
        /
        "payload_registry.jsonl"
    )

    if not family_path.exists():

        raise FileNotFoundError(
            family_path
        )

    if not payload_path.exists():

        raise FileNotFoundError(
            payload_path
        )

    print(
        "Loading family registry..."
    )

    family_registry = read_json(
        family_path
    )

    print(
        "Loading payload registry..."
    )

    payloads = read_jsonl(
        payload_path
    )

    templates = {}

    for split, file_name in (
        TEMPLATE_FILES.items()
    ):

        path = (
            registry_dir
            /
            file_name
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        templates[
            split
        ] = read_jsonl(
            path
        )

    errors = []

    print(
        "Validating taxonomy..."
    )

    taxonomy = (
        validate_family_registry(
            family_registry,
            errors,
        )
    )

    print(
        "Validating payloads..."
    )

    if (
        len(
            payloads
        )
        != EXPECTED_PAYLOAD_COUNT
    ):

        add_error(
            errors,
            "payload_count_mismatch",
            file_name="payload_registry.jsonl",
            details={
                "expected":
                    EXPECTED_PAYLOAD_COUNT,

                "actual":
                    len(
                        payloads
                    ),
            },
        )

    payload_index = (
        validate_payload_registry(
            payloads,
            taxonomy,
            errors,
        )
    )

    print(
        "Validating templates..."
    )

    template_index = (
        validate_template_registry(
            templates,
            taxonomy,
            payload_index,
            errors,
        )
    )

    print(
        "Validating contamination audit..."
    )

    contamination_report = (
        validate_contamination_report(
            registry_dir,
            errors,
        )
    )

    template_counts = {
        split:
            len(
                records
            )

        for split, records
        in templates.items()
    }

    cluster_counts = {}

    for split in TEMPLATE_FILES:

        cluster_counts[
            split
        ] = len(
            {
                cluster

                for cluster, split_set
                in template_index[
                    "cluster_splits"
                ].items()

                if split in split_set
            }
        )

    family_counts = {}

    for split in TEMPLATE_FILES:

        family_counts[
            split
        ] = dict(
            sorted(
                template_index[
                    "family_split_counts"
                ][
                    split
                ].items()
            )
        )

    error_summary = Counter(
        error.get(
            "reason",
            "unknown",
        )
        for error
        in errors
    )

    status = (
        "PASS"
        if not errors
        else "FAIL"
    )

    report = {
        "status":
            status,

        "registry_version":
            family_registry.get(
                "registry_version"
            ),

        "taxonomy_version":
            family_registry.get(
                "taxonomy_version"
            ),

        "errors":
            len(
                errors
            ),

        "warnings":
            0,

        "template_counts":
            template_counts,

        "total_templates":
            sum(
                template_counts.values()
            ),

        "payload_count":
            len(
                payloads
            ),

        "semantic_cluster_counts":
            cluster_counts,

        "total_semantic_clusters":
            len(
                template_index[
                    "clusters"
                ]
            ),

        "family_counts":
            family_counts,

        "development_families":
            sorted(
                taxonomy[
                    "development_families"
                ]
            ),

        "ood_families":
            sorted(
                taxonomy[
                    "ood_families"
                ]
            ),

        "contamination_check": {
            "status":
                contamination_report.get(
                    "status"
                ),

            "blocking_matches":
                contamination_report.get(
                    "blocking_matches"
                ),

            "semantic_check":
                contamination_report.get(
                    "semantic_check"
                ),
        },

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

    report_path = (
        registry_dir
        /
        "validation_report.json"
    )

    error_path = (
        registry_dir
        /
        "validation_errors.jsonl"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_jsonl(
        errors,
        error_path,
    )

    print(
        "\n================================"
    )

    print(
        "Template Registry Validation"
    )

    print(
        "================================"
    )

    print(
        "\nTemplates:"
    )

    for split in TEMPLATE_FILES:

        print(
            f"  {split}: "
            f"{template_counts[split]}"
        )

    print(
        "  total:",
        sum(
            template_counts.values()
        ),
    )

    print(
        "\nPayloads:",
        len(
            payloads
        ),
    )

    print(
        "\nSemantic clusters:"
    )

    for split in TEMPLATE_FILES:

        print(
            f"  {split}: "
            f"{cluster_counts[split]}"
        )

    print(
        "  total:",
        len(
            template_index[
                "clusters"
            ]
        ),
    )

    print(
        "\nContamination:"
    )

    print(
        "  status:",
        contamination_report.get(
            "status"
        ),
    )

    print(
        "  blocking matches:",
        contamination_report.get(
            "blocking_matches"
        ),
    )

    semantic_check = (
        contamination_report.get(
            "semantic_check",
            {},
        )
    )

    print(
        "  semantic audit:",
        semantic_check.get(
            "ran"
        ),
    )

    print(
        "\nErrors:",
        len(
            errors
        ),
    )

    if error_summary:

        print(
            "Error summary:"
        )

        for reason, count in sorted(
            error_summary.items()
        ):

            print(
                f"  {reason}: {count}"
            )

    print(
        "Status:",
        status,
    )

    print(
        "\nReport:",
        report_path,
    )

    if errors:

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()