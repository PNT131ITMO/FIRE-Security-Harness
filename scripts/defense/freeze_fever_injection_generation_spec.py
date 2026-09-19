from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


EXPECTED_TRACKS = {
    "train": {
        "base_split": "train",
        "template_split": "train",
        "expected_records": 31478,
    },
    "validation_seen": {
        "base_split": "paper_dev",
        "template_split": "train",
        "expected_records": 3075,
    },
    "validation_unseen": {
        "base_split": "paper_dev",
        "template_split": "validation",
        "expected_records": 3075,
    },
    "paper_test_seen": {
        "base_split": "paper_test",
        "template_split": "train",
        "expected_records": 3393,
    },
    "paper_test_ood_template": {
        "base_split": "paper_test",
        "template_split": "ood_template",
        "expected_records": 3393,
    },
    "paper_test_ood_family": {
        "base_split": "paper_test",
        "template_split": "ood_family",
        "expected_records": 3393,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def count_jsonl(path: Path) -> int:
    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                count += 1

    return count


def add_error(
    errors: list[dict],
    reason: str,
    details=None,
) -> None:
    item = {
        "reason": reason,
    }

    if details is not None:
        item["details"] = details

    errors.append(item)


def validate_tracks(
    config: dict,
    errors: list[dict],
) -> None:
    tracks = config.get(
        "tracks",
        {},
    )

    if set(tracks) != set(EXPECTED_TRACKS):
        add_error(
            errors,
            "track_set_mismatch",
            {
                "expected": sorted(EXPECTED_TRACKS),
                "actual": sorted(tracks),
            },
        )
        return

    for name, expected in EXPECTED_TRACKS.items():
        actual = tracks[name]

        for field in (
            "base_split",
            "template_split",
            "expected_records",
        ):
            if (
                actual.get(field)
                != expected[field]
            ):
                add_error(
                    errors,
                    "track_mapping_mismatch",
                    {
                        "track": name,
                        "field": field,
                        "expected": expected[field],
                        "actual": actual.get(field),
                    },
                )

    for name in (
        "paper_test_seen",
        "paper_test_ood_template",
        "paper_test_ood_family",
    ):
        track = tracks[name]

        for field in (
            "allow_training",
            "allow_model_selection",
            "allow_threshold_tuning",
        ):
            if track.get(field) is not False:
                add_error(
                    errors,
                    "paper_test_policy_violation",
                    {
                        "track": name,
                        "field": field,
                    },
                )

        if track.get(
            "final_evaluation"
        ) is not True:
            add_error(
                errors,
                "paper_test_not_final_evaluation",
                name,
            )


def validate_render_plan(
    config: dict,
    errors: list[dict],
) -> None:
    render_plan = config.get(
        "render_plan",
        {},
    )

    shared_groups = render_plan.get(
        "shared_groups",
        {},
    )

    expected_dev = {
        "validation_seen",
        "validation_unseen",
    }

    actual_dev = set(
        shared_groups.get(
            "paper_dev",
            {},
        ).get(
            "tracks",
            [],
        )
    )

    if actual_dev != expected_dev:
        add_error(
            errors,
            "paper_dev_render_group_mismatch",
        )

    expected_test = {
        "paper_test_seen",
        "paper_test_ood_template",
        "paper_test_ood_family",
    }

    actual_test = set(
        shared_groups.get(
            "paper_test",
            {},
        ).get(
            "tracks",
            [],
        )
    )

    if actual_test != expected_test:
        add_error(
            errors,
            "paper_test_render_group_mismatch",
        )

    position = render_plan.get(
        "position",
        {},
    )

    singleton = set(
        position.get(
            "singleton_context",
            {},
        ).get(
            "eligible",
            [],
        )
    )

    if singleton != {
        "prefix",
        "suffix",
    }:
        add_error(
            errors,
            "singleton_position_policy_invalid",
        )

    multi = set(
        position.get(
            "multi_sentence_context",
            {},
        ).get(
            "eligible",
            [],
        )
    )

    if multi != {
        "prefix",
        "middle",
        "suffix",
    }:
        add_error(
            errors,
            "multi_sentence_position_policy_invalid",
        )

    if (
        position.get(
            "ineligible_requested_position",
            {},
        ).get(
            "record_adjustment"
        )
        is not True
    ):
        add_error(
            errors,
            "position_adjustment_not_recorded",
        )


def validate_label_policy(
    config: dict,
    errors: list[dict],
) -> None:
    generation = config.get(
        "generation",
        {},
    )

    gold_access = generation.get(
        "assignment",
        {},
    ).get(
        "gold_label_access",
        {},
    )

    for field in (
        "context_generation",
        "template_assignment",
        "payload_assignment",
        "target_assignment",
    ):
        if gold_access.get(field) is not False:
            add_error(
                errors,
                "gold_label_access_in_context_generation",
                field,
            )

    targeted = config.get(
        "fire_evaluation",
        {},
    ).get(
        "targeted_asr",
        {},
    )

    if targeted.get(
        "build_separate_subset"
    ) is not True:
        add_error(
            errors,
            "targeted_asr_not_separate",
        )

    if set(
        targeted.get(
            "objectives",
            [],
        )
    ) != {
        "force_true",
        "force_false",
    }:
        add_error(
            errors,
            "targeted_asr_objectives_invalid",
        )

    if targeted.get(
        "allow_training"
    ) is not False:
        add_error(
            errors,
            "targeted_asr_training_allowed",
        )

    if targeted.get(
        "allow_model_selection"
    ) is not False:
        add_error(
            errors,
            "targeted_asr_model_selection_allowed",
        )

    if targeted.get(
        "allow_threshold_tuning"
    ) is not False:
        add_error(
            errors,
            "targeted_asr_threshold_tuning_allowed",
        )


def validate_markers(
    config: dict,
    errors: list[dict],
) -> None:
    markers = set(
        config.get(
            "rendering",
            {},
        ).get(
            "forbidden_literal_markers",
            [],
        )
    )

    required = {
        "[ATTACK]",
        "[INJECTION]",
        "<ATTACK>",
        "<INJECTION>",
    }

    if not required.issubset(markers):
        add_error(
            errors,
            "forbidden_marker_policy_incomplete",
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/defense/"
            "fever_injection_generation_v1.yaml"
        ),
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    if not config_path.exists():
        raise FileNotFoundError(
            config_path
        )

    config = read_yaml(
        config_path
    )

    errors = []

    if config.get(
        "spec_version"
    ) != "1.0":
        add_error(
            errors,
            "invalid_spec_version",
        )

    if config.get(
        "generator_version"
    ) != "1.0":
        add_error(
            errors,
            "invalid_generator_version",
        )

    if config.get(
        "generation_seed"
    ) != 42:
        add_error(
            errors,
            "generation_seed_mismatch",
        )

    fever = config[
        "inputs"
    ][
        "fever_benign"
    ]

    fever_manifest_path = Path(
        fever[
            "manifest"
        ]
    )

    fever_validation_path = Path(
        fever[
            "validation_report"
        ]
    )

    if not fever_manifest_path.exists():
        add_error(
            errors,
            "fever_manifest_missing",
        )

    if not fever_validation_path.exists():
        add_error(
            errors,
            "fever_validation_report_missing",
        )

    fever_manifest = {}

    if fever_manifest_path.exists():
        fever_manifest = read_json(
            fever_manifest_path
        )

        if (
            fever_manifest.get(
                "dataset_version"
            )
            != fever.get(
                "dataset_version"
            )
        ):
            add_error(
                errors,
                "fever_dataset_version_mismatch",
            )

        if (
            fever_manifest.get(
                "primary_granularity"
            )
            != "context"
        ):
            add_error(
                errors,
                "fever_primary_granularity_not_context",
            )

        paper_test_policy = (
            fever_manifest.get(
                "paper_test_policy",
                {},
            )
        )

        if (
            paper_test_policy.get(
                "locked"
            )
            is not True
        ):
            add_error(
                errors,
                "fever_paper_test_not_locked",
            )

        if (
            paper_test_policy.get(
                "allow_training"
            )
            is not False
        ):
            add_error(
                errors,
                "fever_paper_test_training_allowed",
            )

    fever_validation = {}

    if fever_validation_path.exists():
        fever_validation = read_json(
            fever_validation_path
        )

        if (
            fever_validation.get(
                "status"
            )
            != "PASS"
        ):
            add_error(
                errors,
                "fever_step3b_validation_not_pass",
            )

        if (
            fever_validation.get(
                "errors"
            )
            != 0
        ):
            add_error(
                errors,
                "fever_step3b_validation_errors",
            )

    input_files = {}

    for split, info in (
        fever[
            "splits"
        ].items()
    ):
        path = Path(
            info[
                "path"
            ]
        )

        if not path.exists():
            add_error(
                errors,
                "fever_context_file_missing",
                {
                    "split": split,
                    "path": str(path),
                },
            )
            continue

        records = count_jsonl(
            path
        )

        expected = info[
            "expected_records"
        ]

        if records != expected:
            add_error(
                errors,
                "fever_context_count_mismatch",
                {
                    "split": split,
                    "expected": expected,
                    "actual": records,
                },
            )

        input_files[
            split
        ] = {
            "path": str(path.as_posix()),
            "records": records,
            "sha256": sha256_file(path),
        }

    registry = config[
        "inputs"
    ][
        "attack_registry"
    ]

    registry_manifest_path = Path(
        registry[
            "manifest"
        ]
    )

    if not registry_manifest_path.exists():
        add_error(
            errors,
            "registry_manifest_missing",
        )

        registry_manifest = {}

    else:
        registry_manifest = read_json(
            registry_manifest_path
        )

        if (
            registry_manifest.get(
                "frozen"
            )
            is not True
        ):
            add_error(
                errors,
                "registry_not_frozen",
            )

        actual_registry_sha = (
            registry_manifest.get(
                "registry_sha256"
            )
        )

        expected_registry_sha = (
            registry.get(
                "registry_sha256"
            )
        )

        if (
            actual_registry_sha
            != expected_registry_sha
        ):
            add_error(
                errors,
                "registry_sha256_mismatch",
                {
                    "expected":
                        expected_registry_sha,
                    "actual":
                        actual_registry_sha,
                },
            )

    validate_tracks(
        config,
        errors,
    )

    validate_render_plan(
        config,
        errors,
    )

    validate_label_policy(
        config,
        errors,
    )

    validate_markers(
        config,
        errors,
    )

    expected_total = sum(
        track[
            "expected_records"
        ]
        for track in EXPECTED_TRACKS.values()
    )

    configured_total = (
        config.get(
            "expected",
            {},
        ).get(
            "total_context_records"
        )
    )

    if (
        configured_total
        != expected_total
    ):
        add_error(
            errors,
            "expected_total_context_records_mismatch",
            {
                "expected": expected_total,
                "configured": configured_total,
            },
        )

    config_sha256 = sha256_file(
        config_path
    )

    output_root = Path(
        config[
            "output"
        ][
            "root"
        ]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_report = {
        "step": "3D.1",
        "spec_name": config.get(
            "spec_name"
        ),
        "spec_version": config.get(
            "spec_version"
        ),
        "generation_config_sha256":
            config_sha256,
        "errors": len(errors),
        "status":
            "PASS"
            if not errors
            else "FAIL",
        "error_details": errors,
    }

    validation_report_path = (
        output_root
        /
        "spec_validation_report.json"
    )

    validation_report_path.write_text(
        json.dumps(
            validation_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n================================"
    )
    print(
        "FEVER Injection Generation Spec"
    )
    print(
        "================================"
    )

    print(
        "\nSpec version:",
        config.get(
            "spec_version"
        ),
    )

    print(
        "Generator version:",
        config.get(
            "generator_version"
        ),
    )

    print(
        "Generation seed:",
        config.get(
            "generation_seed"
        ),
    )

    print(
        "\nRegistry SHA-256:",
        registry.get(
            "registry_sha256"
        ),
    )

    print(
        "\nInput FEVER contexts:"
    )

    for split in (
        "train",
        "paper_dev",
        "paper_test",
    ):
        info = input_files.get(
            split,
            {},
        )

        print(
            f"  {split}:",
            info.get(
                "records",
                "missing",
            ),
        )

    print(
        "\nPlanned context outputs:",
        expected_total,
    )

    print(
        "Errors:",
        len(errors),
    )

    print(
        "Status:",
        validation_report[
            "status"
        ],
    )

    if errors:
        print(
            "\nReport:",
            validation_report_path,
        )

        raise SystemExit(1)

    manifest = {
        "artifact":
            "FEVER Injection Generation Specification",

        "step":
            "3D.1",

        "spec_name":
            config[
                "spec_name"
            ],

        "spec_version":
            config[
                "spec_version"
            ],

        "generator_version":
            config[
                "generator_version"
            ],

        "generation_seed":
            config[
                "generation_seed"
            ],

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
                config_sha256,
        },

        "inputs": {
            "fever_benign_manifest": {
                "path":
                    str(
                        fever_manifest_path.as_posix()
                    ),

                "sha256":
                    sha256_file(
                        fever_manifest_path
                    ),
            },

            "fever_validation_report": {
                "path":
                    str(
                        fever_validation_path.as_posix()
                    ),

                "sha256":
                    sha256_file(
                        fever_validation_path
                    ),
            },

            "fever_contexts":
                input_files,

            "attack_registry_manifest": {
                "path":
                    str(
                        registry_manifest_path.as_posix()
                    ),

                "sha256":
                    sha256_file(
                        registry_manifest_path
                    ),

                "registry_sha256":
                    registry_manifest.get(
                        "registry_sha256"
                    ),
            },
        },

        "track_mapping":
            {
                name: {
                    "base_split":
                        track[
                            "base_split"
                        ],

                    "template_split":
                        track[
                            "template_split"
                        ],

                    "expected_records":
                        track[
                            "expected_records"
                        ],
                }
                for name, track
                in EXPECTED_TRACKS.items()
            },

        "expected_total_context_records":
            expected_total,

        "policies": {
            "primary_granularity":
                "context",

            "clean_injected_pairing":
                "same_base_evidence_context",

            "evidence_modification":
                False,

            "llm_generation":
                False,

            "gold_label_used_for_context_generation":
                False,

            "paper_test_training":
                False,

            "paper_test_model_selection":
                False,

            "shared_paper_test_render_plan":
                True,

            "singleton_middle_allowed":
                False,

            "attack_marker_added":
                False,

            "targeted_asr":
                "separate_claim_level_evaluation_subset",
        },

        "validation": {
            "status":
                "PASS",

            "errors":
                0,

            "report":
                str(
                    validation_report_path.as_posix()
                ),
        },

        "next_stage":
            "3D.2_deterministic_assignment_and_render_plans",
    }

    manifest_path = (
        output_root
        /
        "spec_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Frozen:",
        True,
    )

    print(
        "Generation config SHA-256:",
        config_sha256,
    )

    print(
        "\nManifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()