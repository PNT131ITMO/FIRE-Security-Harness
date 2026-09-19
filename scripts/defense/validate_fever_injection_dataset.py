from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import build_fever_fire_evaluation_instances as fire
import generate_fever_injected_contexts as gen

TRACKS = (
    "train",
    "validation_seen",
    "validation_unseen",
    "paper_test_seen",
    "paper_test_ood_template",
    "paper_test_ood_family",
)

EVAL_TRACKS = TRACKS[1:]

FORBIDDEN_ASSIGNMENT_FIELDS = {
    "claim",
    "claim_id",
    "claim_ids",
    "fever_label",
    "fever_labels",
    "fire_label",
    "gold_fever_label",
    "gold_fire_label",
    "label",
}


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    value: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def add(
    errors: list[dict],
    reason: str,
    details=None,
) -> None:
    item = {
        "reason": reason,
    }

    if details is not None:
        item[
            "details"
        ] = details

    errors.append(
        item
    )


def restore_clean(
    row: dict,
) -> str:
    return (
        row[
            "text"
        ][
            :row[
                "attack_start_char"
            ]
        ]
        +
        row[
            "text"
        ][
            row[
                "attack_end_char"
            ]:
        ]
    )


def check_prerequisites(
    config: dict,
    config_sha: str,
    errors: list[dict],
) -> None:
    root = Path(
        config[
            "output"
        ][
            "root"
        ]
    )

    spec = read_json(
        root
        /
        "spec_manifest.json"
    )

    plan = read_json(
        root
        /
        "plan_validation_report.json"
    )

    generation = read_json(
        Path(
            config[
                "output"
            ][
                "generation_report"
            ]
        )
    )

    fire_report = read_json(
        root
        /
        "fire_eval_report.json"
    )

    registry_manifest = read_json(
        Path(
            config[
                "inputs"
            ][
                "attack_registry"
            ][
                "manifest"
            ]
        )
    )

    if (
        spec.get(
            "frozen"
        )
        is not True
    ):
        add(
            errors,
            "spec_not_frozen",
        )

    if (
        spec.get(
            "generation_config",
            {},
        ).get(
            "sha256"
        )
        != config_sha
    ):
        add(
            errors,
            "spec_config_sha_mismatch",
        )

    if (
        plan.get(
            "status"
        )
        != "PASS"
        or
        plan.get(
            "errors"
        )
        != 0
    ):
        add(
            errors,
            "plan_validation_not_pass",
        )

    if (
        generation.get(
            "status"
        )
        != "PASS"
    ):
        add(
            errors,
            "generation_report_not_pass",
        )

    if (
        fire_report.get(
            "status"
        )
        != "PASS"
    ):
        add(
            errors,
            "fire_eval_report_not_pass",
        )

    if (
        registry_manifest.get(
            "frozen"
        )
        is not True
    ):
        add(
            errors,
            "registry_not_frozen",
        )

    if (
        registry_manifest.get(
            "registry_sha256"
        )
        !=
        config[
            "inputs"
        ][
            "attack_registry"
        ][
            "registry_sha256"
        ]
    ):
        add(
            errors,
            "registry_sha_mismatch",
        )


def validate_contexts(
    config: dict,
    config_sha: str,
    errors: list[dict],
) -> tuple[
    dict,
    dict,
]:
    root = Path(
        config[
            "output"
        ][
            "root"
        ]
    )

    bases = (
        gen.load_base_contexts(
            config
        )
    )

    (
        templates,
        payloads,
    ) = gen.load_registry(
        config
    )

    assignment_dir = (
        root
        /
        "plans"
        /
        "assignments"
    )

    required = set(
        config[
            "context_schema"
        ][
            "required_fields"
        ]
    )

    family_registry = read_json(
        Path(
            config[
                "inputs"
            ][
                "attack_registry"
            ][
                "root"
            ]
        )
        /
        "family_registry.json"
    )

    dev_families = set(
        family_registry[
            "development_families"
        ]
    )

    ood_families = set(
        family_registry[
            "ood_families"
        ]
    )

    contexts_by_track = {}
    stats = {}

    global_sample_ids = set()
    global_text_hashes = {}

    base_split_map = defaultdict(
        set
    )

    for track in TRACKS:
        track_config = (
            config[
                "tracks"
            ][
                track
            ]
        )

        contexts = (
            gen.read_jsonl(
                root
                /
                track_config[
                    "output"
                ]
            )
        )

        assignments = (
            gen.read_jsonl(
                assignment_dir
                /
                f"{track}.jsonl"
            )
        )

        contexts_by_track[
            track
        ] = contexts

        if (
            len(
                contexts
            )
            !=
            track_config[
                "expected_records"
            ]
        ):
            add(
                errors,
                "context_count_mismatch",
                {
                    "track":
                        track,

                    "actual":
                        len(
                            contexts
                        ),

                    "expected":
                        track_config[
                            "expected_records"
                        ],
                },
            )

        if (
            len(
                assignments
            )
            !=
            track_config[
                "expected_records"
            ]
        ):
            add(
                errors,
                "assignment_count_mismatch",
                {
                    "track":
                        track,

                    "actual":
                        len(
                            assignments
                        ),

                    "expected":
                        track_config[
                            "expected_records"
                        ],
                },
            )

        assignment_map = {}

        for assignment in assignments:
            base_id = assignment[
                "base_text_id"
            ]

            if (
                base_id
                in assignment_map
            ):
                add(
                    errors,
                    "duplicate_assignment_base",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

            assignment_map[
                base_id
            ] = assignment

            forbidden = sorted(
                FORBIDDEN_ASSIGNMENT_FIELDS
                .intersection(
                    assignment
                )
            )

            if forbidden:
                add(
                    errors,
                    "assignment_contains_gold_fields",
                    {
                        "track":
                            track,

                        "base":
                            base_id,

                        "fields":
                            forbidden,
                    },
                )

            template = templates.get(
                assignment[
                    "template_id"
                ]
            )

            if template is None:
                add(
                    errors,
                    "unknown_assignment_template",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

                continue

            try:
                gen.validate_assignment_against_registry(
                    assignment,
                    template,
                    payloads,
                )

            except Exception as error:
                add(
                    errors,
                    "assignment_registry_validation_failed",
                    {
                        "track":
                            track,

                        "base":
                            base_id,

                        "error":
                            str(
                                error
                            ),
                    },
                )

        seen_bases = set()
        seen_hashes = set()

        distributions = {
            "family":
                Counter(),

            "attack_objective":
                Counter(),

            "insertion_position":
                Counter(),

            "representation":
                Counter(),
        }

        for row in contexts:
            base_id = row.get(
                "base_text_id"
            )

            missing = sorted(
                required.difference(
                    row
                )
            )

            if missing:
                add(
                    errors,
                    "missing_context_fields",
                    {
                        "track":
                            track,

                        "base":
                            base_id,

                        "fields":
                            missing,
                    },
                )

            if (
                base_id
                in seen_bases
            ):
                add(
                    errors,
                    "duplicate_base_in_track",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

            seen_bases.add(
                base_id
            )

            sample_id = row.get(
                "sample_id"
            )

            if (
                sample_id
                in global_sample_ids
            ):
                add(
                    errors,
                    "duplicate_sample_id",
                    sample_id,
                )

            global_sample_ids.add(
                sample_id
            )

            if (
                row.get(
                    "dataset_split"
                )
                != track
                or
                row.get(
                    "upstream_split"
                )
                !=
                track_config[
                    "base_split"
                ]
            ):
                add(
                    errors,
                    "context_split_mismatch",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

            base_split_map[
                base_id
            ].add(
                row.get(
                    "upstream_split"
                )
            )

            fixed_values = {
                "label":
                    "injection",

                "source":
                    "fever_derived",

                "original_source":
                    "FEVER",

                "synthetic":
                    True,

                "detector_evaluable":
                    True,

                "end_to_end_evaluable":
                    True,
            }

            for (
                field,
                expected,
            ) in fixed_values.items():
                if (
                    row.get(
                        field
                    )
                    != expected
                ):
                    add(
                        errors,
                        "fixed_value_mismatch",
                        {
                            "track":
                                track,

                            "base":
                                base_id,

                            "field":
                                field,
                        },
                    )

            text = row.get(
                "text"
            )

            block = row.get(
                "inserted_block"
            )

            start = row.get(
                "attack_start_char"
            )

            end = row.get(
                "attack_end_char"
            )

            if (
                not isinstance(
                    text,
                    str,
                )
                or
                not isinstance(
                    block,
                    str,
                )
                or
                not isinstance(
                    start,
                    int,
                )
                or
                not isinstance(
                    end,
                    int,
                )
                or
                start < 0
                or
                end < start
                or
                end > len(
                    text
                )
            ):
                add(
                    errors,
                    "invalid_attack_span",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

                continue

            if (
                text[
                    start:end
                ]
                != block
            ):
                add(
                    errors,
                    "attack_span_block_mismatch",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

            clean_text = (
                restore_clean(
                    row
                )
            )

            expected_hashes = {
                "clean_text_sha256":
                    gen.sha256_text(
                        clean_text
                    ),

                "attack_text_sha256":
                    gen.sha256_text(
                        row[
                            "attack_text"
                        ]
                    ),

                "inserted_block_sha256":
                    gen.sha256_text(
                        block
                    ),

                "injected_text_sha256":
                    gen.sha256_text(
                        text
                    ),

                "text_sha256":
                    gen.sha256_text(
                        text
                    ),
            }

            for (
                field,
                expected,
            ) in expected_hashes.items():
                if (
                    row.get(
                        field
                    )
                    != expected
                ):
                    add(
                        errors,
                        "hash_mismatch",
                        {
                            "track":
                                track,

                            "base":
                                base_id,

                            "field":
                                field,
                        },
                    )

            if (
                row.get(
                    "clean_insertion_offset"
                )
                != start
            ):
                add(
                    errors,
                    "clean_insertion_offset_mismatch",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

            injected_hash = (
                expected_hashes[
                    "injected_text_sha256"
                ]
            )

            if (
                injected_hash
                in seen_hashes
            ):
                add(
                    errors,
                    "duplicate_injected_text_within_track",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

            seen_hashes.add(
                injected_hash
            )

            if (
                injected_hash
                in global_text_hashes
            ):
                add(
                    errors,
                    "duplicate_injected_text_across_tracks",
                    {
                        "first":
                            global_text_hashes[
                                injected_hash
                            ],

                        "second":
                            [
                                track,
                                base_id,
                            ],
                    },
                )

            else:
                global_text_hashes[
                    injected_hash
                ] = [
                    track,
                    base_id,
                ]

            for marker in (
                config[
                    "rendering"
                ][
                    "forbidden_literal_markers"
                ]
            ):
                if marker in text:
                    add(
                        errors,
                        "forbidden_marker",
                        {
                            "track":
                                track,

                            "base":
                                base_id,

                            "marker":
                                marker,
                        },
                    )

            base_record = (
                bases[
                    track_config[
                        "base_split"
                    ]
                ].get(
                    base_id
                )
            )

            if base_record is None:
                add(
                    errors,
                    "base_context_missing",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

                continue

            provenance_checks = (
                (
                    "clean_text",
                    clean_text,
                    base_record[
                        "text"
                    ],
                ),

                (
                    "base_text_sha256",
                    row.get(
                        "base_text_sha256"
                    ),
                    base_record.get(
                        "base_text_sha256"
                    ),
                ),

                (
                    "claim_ids",
                    row.get(
                        "claim_ids"
                    ),
                    base_record.get(
                        "claim_ids"
                    ),
                ),

                (
                    "fever_labels",
                    row.get(
                        "fever_labels"
                    ),
                    base_record.get(
                        "fever_labels"
                    ),
                ),

                (
                    "evidence_refs",
                    row.get(
                        "evidence_refs"
                    ),
                    base_record.get(
                        "evidence_refs"
                    ),
                ),

                (
                    "context_size",
                    row.get(
                        "context_size"
                    ),
                    base_record.get(
                        "context_size"
                    ),
                ),
            )

            for (
                field,
                actual,
                expected,
            ) in provenance_checks:
                if (
                    actual
                    != expected
                ):
                    add(
                        errors,
                        "base_provenance_mismatch",
                        {
                            "track":
                                track,

                            "base":
                                base_id,

                            "field":
                                field,
                        },
                    )

            assignment = (
                assignment_map.get(
                    base_id
                )
            )

            if assignment is None:
                add(
                    errors,
                    "assignment_missing",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

            else:
                template = templates[
                    assignment[
                        "template_id"
                    ]
                ]

                expected_record = (
                    gen.build_sample(
                        base_record,
                        assignment,
                        template,
                        config,
                        config_sha,
                    )
                )

                if (
                    row
                    != expected_record
                ):
                    different_fields = sorted(
                        key
                        for key in (
                            set(
                                row
                            )
                            |
                            set(
                                expected_record
                            )
                        )
                        if (
                            row.get(
                                key
                            )
                            !=
                            expected_record.get(
                                key
                            )
                        )
                    )

                    add(
                        errors,
                        "context_rebuild_mismatch",
                        {
                            "track":
                                track,

                            "base":
                                base_id,

                            "fields":
                                different_fields,
                        },
                    )

            family = row.get(
                "family"
            )

            if (
                track
                in {
                    "train",
                    "validation_seen",
                    "validation_unseen",
                }
                and
                family in ood_families
            ):
                add(
                    errors,
                    "ood_family_in_development",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

            if (
                track
                == "paper_test_ood_family"
                and
                family
                not in ood_families
            ):
                add(
                    errors,
                    "non_ood_family_in_ood_track",
                    {
                        "base":
                            base_id,

                        "family":
                            family,
                    },
                )

            if (
                track
                != "paper_test_ood_family"
                and
                family
                not in dev_families
            ):
                add(
                    errors,
                    "non_development_family",
                    {
                        "track":
                            track,

                        "base":
                            base_id,

                        "family":
                            family,
                    },
                )

            if (
                row.get(
                    "context_size"
                )
                == 1
                and
                row.get(
                    "insertion_position"
                )
                == "middle"
            ):
                add(
                    errors,
                    "singleton_middle",
                    {
                        "track":
                            track,

                        "base":
                            base_id,
                    },
                )

            for field in distributions:
                distributions[
                    field
                ][
                    str(
                        row.get(
                            field
                        )
                    )
                ] += 1

        stats[
            track
        ] = {
            "records":
                len(
                    contexts
                ),

            **{
                field:
                    dict(
                        sorted(
                            counter.items()
                        )
                    )

                for (
                    field,
                    counter,
                ) in distributions.items()
            },
        }

    for (
        base_id,
        splits,
    ) in base_split_map.items():
        if (
            len(
                splits
            )
            != 1
        ):
            add(
                errors,
                "cross_base_split_leakage",
                {
                    "base":
                        base_id,

                    "splits":
                        sorted(
                            splits
                        ),
                },
            )

    for (
        group_name,
        group,
    ) in (
        config[
            "render_plan"
        ][
            "shared_groups"
        ].items()
    ):
        tracks = group[
            "tracks"
        ]

        reference = {
            record[
                "base_text_id"
            ]: (
                record[
                    "render_plan_id"
                ],
                record[
                    "insertion_position"
                ],
                record[
                    "representation"
                ],
                record[
                    "encoding"
                ],
            )

            for record in (
                contexts_by_track[
                    tracks[0]
                ]
            )
        }

        for track in tracks[
            1:
        ]:
            current = {
                record[
                    "base_text_id"
                ]: (
                    record[
                        "render_plan_id"
                    ],
                    record[
                        "insertion_position"
                    ],
                    record[
                        "representation"
                    ],
                    record[
                        "encoding"
                    ],
                )

                for record in (
                    contexts_by_track[
                        track
                    ]
                )
            }

            if (
                current
                != reference
            ):
                add(
                    errors,
                    "shared_render_plan_mismatch",
                    {
                        "group":
                            group_name,

                        "track":
                            track,
                    },
                )

    return (
        contexts_by_track,
        stats,
    )


def validate_fire(
    config: dict,
    config_sha: str,
    contexts: dict,
    errors: list[dict],
) -> dict:
    root = Path(
        config[
            "output"
        ][
            "fire_eval_dir"
        ]
    )

    targeted_root = (
        root
        /
        "targeted_asr"
    )

    gold = (
        fire.load_gold_index(
            config
        )
    )

    stats = {}
    global_ids = set()

    train_file = (
        root
        /
        "train.jsonl"
    )

    if (
        train_file.exists()
        and
        train_file.stat().st_size > 0
    ):
        add(
            errors,
            "unexpected_train_fire_eval_file",
            str(
                train_file
            ),
        )

    for track in EVAL_TRACKS:
        track_config = (
            config[
                "tracks"
            ][
                track
            ]
        )

        full = (
            fire.read_jsonl(
                root
                /
                f"{track}.jsonl"
            )
        )

        targeted = (
            fire.read_jsonl(
                targeted_root
                /
                f"{track}.jsonl"
            )
        )

        context_map = {
            record[
                "sample_id"
            ]:
                record

            for record in (
                contexts[
                    track
                ]
            )
        }

        expected = {}

        for context in (
            context_map.values()
        ):
            clean_text = (
                fire.restore_clean_text(
                    context
                )
            )

            for raw_claim_id in (
                context[
                    "claim_ids"
                ]
            ):
                claim_id = (
                    fire.canonical_claim_id(
                        raw_claim_id
                    )
                )

                gold_record = (
                    gold[
                        track_config[
                            "base_split"
                        ]
                    ].get(
                        claim_id
                    )
                )

                if gold_record is None:
                    add(
                        errors,
                        "gold_claim_missing",
                        {
                            "track":
                                track,

                            "claim_id":
                                claim_id,
                        },
                    )

                    continue

                try:
                    fire.validate_context_claim_provenance(
                        context,
                        gold_record,
                    )

                    expected_record = (
                        fire.build_evaluation_instance(
                            track,
                            track_config,
                            context,
                            gold_record,
                            clean_text,
                            config_sha,
                        )
                    )

                except Exception as error:
                    add(
                        errors,
                        "fire_instance_rebuild_failed",
                        {
                            "track":
                                track,

                            "claim_id":
                                claim_id,

                            "error":
                                str(
                                    error
                                ),
                        },
                    )

                    continue

                expected[
                    expected_record[
                        "evaluation_instance_id"
                    ]
                ] = expected_record

        actual = {}

        for record in full:
            instance_id = record[
                "evaluation_instance_id"
            ]

            if (
                instance_id
                in global_ids
            ):
                add(
                    errors,
                    "duplicate_fire_instance_id",
                    instance_id,
                )

            global_ids.add(
                instance_id
            )

            if (
                instance_id
                in actual
            ):
                add(
                    errors,
                    "duplicate_fire_instance_in_track",
                    {
                        "track":
                            track,

                        "id":
                            instance_id,
                    },
                )

            actual[
                instance_id
            ] = record

        if (
            set(
                actual
            )
            !=
            set(
                expected
            )
        ):
            add(
                errors,
                "fire_instance_set_mismatch",
                {
                    "track":
                        track,

                    "expected":
                        len(
                            expected
                        ),

                    "actual":
                        len(
                            actual
                        ),

                    "missing":
                        len(
                            set(
                                expected
                            )
                            -
                            set(
                                actual
                            )
                        ),

                    "extra":
                        len(
                            set(
                                actual
                            )
                            -
                            set(
                                expected
                            )
                        ),
                },
            )

        for instance_id in (
            set(
                actual
            )
            &
            set(
                expected
            )
        ):
            if (
                actual[
                    instance_id
                ]
                !=
                expected[
                    instance_id
                ]
            ):
                different_fields = sorted(
                    key
                    for key in (
                        set(
                            actual[
                                instance_id
                            ]
                        )
                        |
                        set(
                            expected[
                                instance_id
                            ]
                        )
                    )
                    if (
                        actual[
                            instance_id
                        ].get(
                            key
                        )
                        !=
                        expected[
                            instance_id
                        ].get(
                            key
                        )
                    )
                )

                add(
                    errors,
                    "fire_instance_rebuild_mismatch",
                    {
                        "track":
                            track,

                        "id":
                            instance_id,

                        "fields":
                            different_fields,
                    },
                )

        expected_targeted = {
            instance_id

            for (
                instance_id,
                record,
            ) in expected.items()

            if (
                record[
                    "targeted_asr_evaluable"
                ]
                is True
            )
        }

        targeted_map = {
            record[
                "evaluation_instance_id"
            ]:
                record

            for record in targeted
        }

        if (
            set(
                targeted_map
            )
            != expected_targeted
        ):
            add(
                errors,
                "targeted_subset_mismatch",
                {
                    "track":
                        track,

                    "expected":
                        len(
                            expected_targeted
                        ),

                    "actual":
                        len(
                            targeted_map
                        ),
                },
            )

        for (
            instance_id,
            record,
        ) in targeted_map.items():
            if (
                instance_id
                in actual
                and
                record
                != actual[
                    instance_id
                ]
            ):
                add(
                    errors,
                    "targeted_record_differs_from_full",
                    {
                        "track":
                            track,

                        "id":
                            instance_id,
                    },
                )

        stats[
            track
        ] = {
            "contexts":
                len(
                    context_map
                ),

            "claim_instances":
                len(
                    full
                ),

            "targeted_asr_instances":
                len(
                    targeted
                ),
        }

    return stats


def static_validate(
    config_path: Path,
) -> tuple[
    dict,
    dict,
    list[dict],
]:
    config = gen.read_yaml(
        config_path
    )

    config_sha = (
        gen.sha256_file(
            config_path
        )
    )

    errors = []

    check_prerequisites(
        config,
        config_sha,
        errors,
    )

    (
        contexts,
        context_stats,
    ) = validate_contexts(
        config,
        config_sha,
        errors,
    )

    fire_stats = validate_fire(
        config,
        config_sha,
        contexts,
        errors,
    )

    total_context = sum(
        value[
            "records"
        ]
        for value
        in context_stats.values()
    )

    total_fire = sum(
        value[
            "claim_instances"
        ]
        for value
        in fire_stats.values()
    )

    total_targeted = sum(
        value[
            "targeted_asr_instances"
        ]
        for value
        in fire_stats.values()
    )

    if (
        total_context
        !=
        config[
            "expected"
        ][
            "total_context_records"
        ]
    ):
        add(
            errors,
            "total_context_count_mismatch",
            {
                "expected":
                    config[
                        "expected"
                    ][
                        "total_context_records"
                    ],

                "actual":
                    total_context,
            },
        )

    stats = {
        "context_tracks":
            context_stats,

        "fire_eval_tracks":
            fire_stats,

        "total_context_records":
            total_context,

        "total_fire_eval_instances":
            total_fire,

        "total_targeted_asr_instances":
            total_targeted,
    }

    return (
        config,
        stats,
        errors,
    )


def artifact_paths(
    config: dict,
) -> dict[
    str,
    Path,
]:
    root = Path(
        config[
            "output"
        ][
            "root"
        ]
    )

    paths = {
        "render_plans.jsonl":
            Path(
                config[
                    "output"
                ][
                    "render_plans"
                ]
            ),

        "plan_report.json":
            root
            /
            "plan_report.json",

        "plan_validation_report.json":
            root
            /
            "plan_validation_report.json",

        "generation_report.json":
            Path(
                config[
                    "output"
                ][
                    "generation_report"
                ]
            ),

        "fire_eval_report.json":
            root
            /
            "fire_eval_report.json",
    }

    for track in TRACKS:
        paths[
            f"assignments/{track}.jsonl"
        ] = (
            root
            /
            "plans"
            /
            "assignments"
            /
            f"{track}.jsonl"
        )

        paths[
            f"context/{track}.jsonl"
        ] = (
            root
            /
            config[
                "tracks"
            ][
                track
            ][
                "output"
            ]
        )

    fire_root = Path(
        config[
            "output"
        ][
            "fire_eval_dir"
        ]
    )

    for track in EVAL_TRACKS:
        paths[
            f"fire_eval/{track}.jsonl"
        ] = (
            fire_root
            /
            f"{track}.jsonl"
        )

        paths[
            (
                "fire_eval/"
                "targeted_asr/"
                f"{track}.jsonl"
            )
        ] = (
            fire_root
            /
            "targeted_asr"
            /
            f"{track}.jsonl"
        )

    return paths


def snapshot(
    paths: dict[
        str,
        Path,
    ],
) -> dict:
    result = {}

    for (
        name,
        path,
    ) in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                path
            )

        result[
            name
        ] = {
            "path":
                str(
                    path.as_posix()
                ),

            "sha256":
                gen.sha256_file(
                    path
                ),

            "records":
                (
                    len(
                        gen.read_jsonl(
                            path
                        )
                    )
                    if (
                        path.suffix
                        == ".jsonl"
                    )
                    else None
                ),
        }

    return result


def rebuild(
    config_path: Path,
) -> list[dict]:
    project_root = (
        Path(
            __file__
        )
        .resolve()
        .parents[2]
    )

    script_dir = (
        Path(
            __file__
        )
        .resolve()
        .parent
    )

    scripts = (
        "build_fever_injection_plans.py",
        "validate_fever_injection_plans.py",
        "generate_fever_injected_contexts.py",
        "build_fever_fire_evaluation_instances.py",
    )

    result = []

    for name in scripts:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    script_dir
                    /
                    name
                ),
                "--config",
                str(
                    config_path.resolve()
                ),
            ],
            cwd=project_root,
            text=True,
            capture_output=True,
        )

        result.append(
            {
                "script":
                    name,

                "returncode":
                    completed.returncode,

                "stdout_tail":
                    completed.stdout[
                        -4000:
                    ],

                "stderr_tail":
                    completed.stderr[
                        -4000:
                    ],
            }
        )

        if (
            completed.returncode
            != 0
        ):
            break

    return result


def dataset_hash(
    artifacts: dict,
) -> str:
    digest = hashlib.sha256()

    for name in sorted(
        artifacts
    ):
        digest.update(
            (
                f"{name}\0"
                f"{artifacts[name]['sha256']}\n"
            ).encode(
                "utf-8"
            )
        )

    return digest.hexdigest()


def build_manifest(
    config_path: Path,
    config: dict,
    stats: dict,
    artifacts: dict,
) -> dict:
    script_dir = (
        Path(
            __file__
        )
        .resolve()
        .parent
    )

    script_names = (
        "freeze_fever_injection_generation_spec.py",
        "build_fever_injection_plans.py",
        "validate_fever_injection_plans.py",
        "generate_fever_injected_contexts.py",
        "build_fever_fire_evaluation_instances.py",
        "validate_fever_injection_dataset.py",
    )

    scripts = {}

    for name in script_names:
        path = (
            script_dir
            /
            name
        )

        if path.exists():
            scripts[
                name
            ] = {
                "path":
                    str(
                        path.as_posix()
                    ),

                "sha256":
                    gen.sha256_file(
                        path
                    ),
            }

    return {
        "artifact":
            (
                "FEVER-derived Prompt "
                "Injection Dataset"
            ),

        "step":
            "3D",

        "version":
            config[
                "generator_version"
            ],

        "status":
            "frozen",

        "frozen":
            True,

        "mutable":
            False,

        "generation_config": {
            "path":
                str(
                    config_path.as_posix()
                ),

            "sha256":
                gen.sha256_file(
                    config_path
                ),
        },

        "attack_registry": {
            "version":
                config[
                    "inputs"
                ][
                    "attack_registry"
                ][
                    "version"
                ],

            "registry_sha256":
                config[
                    "inputs"
                ][
                    "attack_registry"
                ][
                    "registry_sha256"
                ],
        },

        "counts": {
            "context_records":
                stats[
                    "total_context_records"
                ],

            "fire_eval_instances":
                stats[
                    "total_fire_eval_instances"
                ],

            "targeted_asr_instances":
                stats[
                    "total_targeted_asr_instances"
                ],
        },

        "policies": {
            "primary_granularity":
                "context",

            "fire_eval_granularity":
                "claim",

            "evidence_rewrite_allowed":
                False,

            "gold_label_used_for_context_generation":
                False,

            "gold_label_used_for_targeted_eval_selection":
                True,

            "paper_test_training_allowed":
                False,

            "paper_test_model_selection_allowed":
                False,

            "shared_render_plan_for_paired_tracks":
                True,

            "deterministic_generation":
                True,
        },

        "artifacts":
            artifacts,

        "dataset_sha256":
            dataset_hash(
                artifacts
            ),

        "pipeline_scripts":
            scripts,

        "next_stage":
            "3E_hard_benign_generation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/defense/"
            "fever_injection_generation_v1.yaml"
        ),
    )

    parser.add_argument(
        "--verify-reproducibility",
        action="store_true",
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    (
        config,
        stats,
        errors,
    ) = static_validate(
        config_path
    )

    root = Path(
        config[
            "output"
        ][
            "root"
        ]
    )

    report_path = Path(
        config[
            "output"
        ][
            "validation_report"
        ]
    )

    error_path = (
        root
        /
        "validation_errors.jsonl"
    )

    report = {
        "step":
            "3D.5",

        "status":
            (
                "PASS"
                if not errors
                else "FAIL"
            ),

        "errors":
            len(
                errors
            ),

        "warnings":
            0,

        "generation_config_sha256":
            gen.sha256_file(
                config_path
            ),

        "registry_sha256":
            config[
                "inputs"
            ][
                "attack_registry"
            ][
                "registry_sha256"
            ],

        "stats":
            stats,

        "error_summary":
            dict(
                sorted(
                    Counter(
                        error[
                            "reason"
                        ]
                        for error
                        in errors
                    ).items()
                )
            ),

        "error_details":
            errors[
                :2000
            ],

        "reproducibility_checked":
            False,
    }

    write_json(
        report_path,
        report,
    )

    write_jsonl(
        error_path,
        errors,
    )

    print(
        "\n================================"
    )

    print(
        "FEVER Injection Dataset Validation"
    )

    print(
        "================================"
    )

    print(
        "\nContext records:",
        stats[
            "total_context_records"
        ],
    )

    print(
        "FIRE instances:",
        stats[
            "total_fire_eval_instances"
        ],
    )

    print(
        "Targeted ASR instances:",
        stats[
            "total_targeted_asr_instances"
        ],
    )

    print(
        "Errors:",
        len(
            errors
        ),
    )

    print(
        "Static status:",
        report[
            "status"
        ],
    )

    if errors:
        print(
            "Report:",
            report_path,
        )

        raise SystemExit(
            1
        )

    if (
        not args.verify_reproducibility
    ):
        print(
            "\nReproducibility: not run"
        )

        print(
            "Run again with "
            "--verify-reproducibility "
            "to complete Step 3D.5"
        )

        return

    paths = artifact_paths(
        config
    )

    before = snapshot(
        paths
    )

    print(
        "\nRebuilding deterministic artifacts..."
    )

    runs = rebuild(
        config_path
    )

    repro_path = (
        root
        /
        "reproducibility_report.json"
    )

    if any(
        run[
            "returncode"
        ]
        != 0

        for run in runs
    ):
        write_json(
            repro_path,
            {
                "step":
                    "3D.5",

                "status":
                    "FAIL",

                "runs":
                    runs,

                "artifacts_checked":
                    len(
                        paths
                    ),

                "mismatches":
                    -1,

                "mismatch_details":
                    [],
            },
        )

        print(
            "Reproducibility: FAIL"
        )

        print(
            "Report:",
            repro_path,
        )

        raise SystemExit(
            1
        )

    (
        config_after,
        stats_after,
        errors_after,
    ) = static_validate(
        config_path
    )

    after = snapshot(
        paths
    )

    mismatches = []

    for name in sorted(
        paths
    ):
        if (
            before[
                name
            ][
                "sha256"
            ]
            !=
            after[
                name
            ][
                "sha256"
            ]
            or
            before[
                name
            ][
                "records"
            ]
            !=
            after[
                name
            ][
                "records"
            ]
        ):
            mismatches.append(
                {
                    "artifact":
                        name,

                    "before_sha256":
                        before[
                            name
                        ][
                            "sha256"
                        ],

                    "after_sha256":
                        after[
                            name
                        ][
                            "sha256"
                        ],

                    "before_records":
                        before[
                            name
                        ][
                            "records"
                        ],

                    "after_records":
                        after[
                            name
                        ][
                            "records"
                        ],
                }
            )

    repro_status = (
        "PASS"
        if (
            not mismatches
            and
            not errors_after
        )
        else "FAIL"
    )

    repro = {
        "step":
            "3D.5",

        "status":
            repro_status,

        "generation_config_sha256":
            gen.sha256_file(
                config_path
            ),

        "runs":
            runs,

        "artifacts_checked":
            len(
                paths
            ),

        "mismatches":
            len(
                mismatches
            ),

        "mismatch_details":
            mismatches,

        "post_rebuild_validation_errors":
            len(
                errors_after
            ),

        "before":
            before,

        "after":
            after,
    }

    write_json(
        repro_path,
        repro,
    )

    final_report = {
        "step":
            "3D.5",

        "status":
            (
                "PASS"
                if (
                    repro_status
                    == "PASS"
                    and
                    not errors_after
                )
                else "FAIL"
            ),

        "errors":
            len(
                errors_after
            ),

        "warnings":
            0,

        "generation_config_sha256":
            gen.sha256_file(
                config_path
            ),

        "registry_sha256":
            config_after[
                "inputs"
            ][
                "attack_registry"
            ][
                "registry_sha256"
            ],

        "stats":
            stats_after,

        "error_summary":
            dict(
                sorted(
                    Counter(
                        error[
                            "reason"
                        ]
                        for error
                        in errors_after
                    ).items()
                )
            ),

        "error_details":
            errors_after[
                :2000
            ],

        "reproducibility_checked":
            True,

        "reproducibility_status":
            repro_status,

        "reproducibility_mismatches":
            len(
                mismatches
            ),
    }

    write_json(
        report_path,
        final_report,
    )

    write_jsonl(
        error_path,
        errors_after,
    )

    if (
        final_report[
            "status"
        ]
        != "PASS"
    ):
        print(
            "\nReproducibility:",
            repro_status,
        )

        print(
            "Mismatches:",
            len(
                mismatches
            ),
        )

        print(
            "Final status: FAIL"
        )

        print(
            "Validation report:",
            report_path,
        )

        print(
            "Reproducibility report:",
            repro_path,
        )

        raise SystemExit(
            1
        )

    manifest = build_manifest(
        config_path,
        config_after,
        stats_after,
        after,
    )

    manifest_path = Path(
        config_after[
            "output"
        ][
            "manifest"
        ]
    )

    write_json(
        manifest_path,
        manifest,
    )

    print(
        "\nReproducibility: PASS"
    )

    print(
        "Artifacts checked:",
        len(
            paths
        ),
    )

    print(
        "Mismatches: 0"
    )

    print(
        "Final status: PASS"
    )

    print(
        "Frozen: True"
    )

    print(
        "Dataset SHA-256:",
        manifest[
            "dataset_sha256"
        ],
    )

    print(
        "\nValidation report:",
        report_path,
    )

    print(
        "Reproducibility report:",
        repro_path,
    )

    print(
        "Manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()