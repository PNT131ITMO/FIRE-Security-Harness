from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


TRACK_ORDER = (
    "train",
    "validation_seen",
    "validation_unseen",
    "paper_test_seen",
    "paper_test_ood_template",
    "paper_test_ood_family",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_jsonl(path: Path) -> list[dict]:
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
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: {error}"
                ) from error

    return records


def add_error(
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

    errors.append(item)


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

    config = read_yaml(
        config_path
    )

    output_root = Path(
        config[
            "output"
        ][
            "root"
        ]
    )

    errors = []

    config_sha256 = sha256_file(
        config_path
    )

    spec_manifest = read_json(
        output_root
        /
        "spec_manifest.json"
    )

    if (
        spec_manifest[
            "generation_config"
        ][
            "sha256"
        ]
        != config_sha256
    ):
        add_error(
            errors,
            "generation_config_sha256_mismatch",
        )

    render_path = Path(
        config[
            "output"
        ][
            "render_plans"
        ]
    )

    render_plans = read_jsonl(
        render_path
    )

    expected_render_plans = sum(
        info[
            "expected_records"
        ]
        for info in config[
            "inputs"
        ][
            "fever_benign"
        ][
            "splits"
        ].values()
    )

    if (
        len(render_plans)
        != expected_render_plans
    ):
        add_error(
            errors,
            "render_plan_count_mismatch",
            {
                "expected":
                    expected_render_plans,

                "actual":
                    len(render_plans),
            },
        )

    render_by_key = {}
    render_ids = set()

    for record in render_plans:
        key = (
            record[
                "base_split"
            ],
            record[
                "base_text_id"
            ],
        )

        if key in render_by_key:
            add_error(
                errors,
                "duplicate_render_plan_base",
                key,
            )

        render_by_key[
            key
        ] = record

        render_plan_id = record[
            "render_plan_id"
        ]

        if render_plan_id in render_ids:
            add_error(
                errors,
                "duplicate_render_plan_id",
                render_plan_id,
            )

        render_ids.add(
            render_plan_id
        )

        context_size = record[
            "context_size"
        ]

        position = record[
            "insertion_position"
        ]

        if (
            context_size == 1
            and position == "middle"
        ):
            add_error(
                errors,
                "singleton_middle_position",
                record[
                    "base_text_id"
                ],
            )

        expected_adjusted = (
            record[
                "requested_position"
            ]
            != record[
                "insertion_position"
            ]
        )

        if (
            record[
                "position_adjusted"
            ]
            != expected_adjusted
        ):
            add_error(
                errors,
                "position_adjusted_mismatch",
                record[
                    "base_text_id"
                ],
            )

        if (
            record[
                "generation_config_sha256"
            ]
            != config_sha256
        ):
            add_error(
                errors,
                "render_plan_config_sha_mismatch",
                record[
                    "base_text_id"
                ],
            )

    assignments_dir = (
        output_root
        /
        "plans"
        /
        "assignments"
    )

    all_assignment_ids = set()

    base_split_usage = defaultdict(
        set
    )

    distributions = {}

    track_records = {}

    for track_name in TRACK_ORDER:
        path = (
            assignments_dir
            /
            f"{track_name}.jsonl"
        )

        if not path.exists():
            add_error(
                errors,
                "assignment_file_missing",
                track_name,
            )

            continue

        records = read_jsonl(
            path
        )

        track_records[
            track_name
        ] = records

        track_config = config[
            "tracks"
        ][
            track_name
        ]

        expected = track_config[
            "expected_records"
        ]

        if len(records) != expected:
            add_error(
                errors,
                "assignment_count_mismatch",
                {
                    "track":
                        track_name,

                    "expected":
                        expected,

                    "actual":
                        len(records),
                },
            )

        base_ids = set()
        family_counts = Counter()
        objective_counts = Counter()
        template_counts = Counter()
        position_counts = Counter()
        representation_counts = Counter()

        for record in records:
            assignment_id = record[
                "assignment_id"
            ]

            if assignment_id in all_assignment_ids:
                add_error(
                    errors,
                    "duplicate_assignment_id",
                    assignment_id,
                )

            all_assignment_ids.add(
                assignment_id
            )

            if (
                record[
                    "dataset_split"
                ]
                != track_name
            ):
                add_error(
                    errors,
                    "dataset_split_mismatch",
                    record[
                        "base_text_id"
                    ],
                )

            if (
                record[
                    "base_split"
                ]
                != track_config[
                    "base_split"
                ]
            ):
                add_error(
                    errors,
                    "base_split_mismatch",
                    {
                        "track":
                            track_name,

                        "base_text_id":
                            record[
                                "base_text_id"
                            ],
                    },
                )

            if (
                record[
                    "template_split"
                ]
                != track_config[
                    "template_split"
                ]
            ):
                add_error(
                    errors,
                    "template_split_mismatch",
                    {
                        "track":
                            track_name,

                        "base_text_id":
                            record[
                                "base_text_id"
                            ],
                    },
                )

            base_text_id = record[
                "base_text_id"
            ]

            if base_text_id in base_ids:
                add_error(
                    errors,
                    "duplicate_base_in_track",
                    {
                        "track":
                            track_name,

                        "base_text_id":
                            base_text_id,
                    },
                )

            base_ids.add(
                base_text_id
            )

            base_split_usage[
                base_text_id
            ].add(
                record[
                    "base_split"
                ]
            )

            render_key = (
                record[
                    "base_split"
                ],
                base_text_id,
            )

            render = render_by_key.get(
                render_key
            )

            if render is None:
                add_error(
                    errors,
                    "render_plan_missing_for_assignment",
                    {
                        "track":
                            track_name,

                        "base_text_id":
                            base_text_id,
                    },
                )

                continue

            for field in (
                "render_plan_id",
                "requested_position",
                "insertion_position",
                "position_adjusted",
                "representation",
                "encoding",
            ):
                if (
                    record[field]
                    != render[field]
                ):
                    add_error(
                        errors,
                        "shared_render_plan_mismatch",
                        {
                            "track":
                                track_name,

                            "base_text_id":
                                base_text_id,

                            "field":
                                field,
                        },
                    )

            if (
                record[
                    "generation_config_sha256"
                ]
                != config_sha256
            ):
                add_error(
                    errors,
                    "assignment_config_sha_mismatch",
                    {
                        "track":
                            track_name,

                        "base_text_id":
                            base_text_id,
                    },
                )

            expected_registry_sha = (
                config[
                    "inputs"
                ][
                    "attack_registry"
                ][
                    "registry_sha256"
                ]
            )

            if (
                record[
                    "registry_sha256"
                ]
                != expected_registry_sha
            ):
                add_error(
                    errors,
                    "assignment_registry_sha_mismatch",
                    {
                        "track":
                            track_name,

                        "base_text_id":
                            base_text_id,
                    },
                )

            family_counts[
                record[
                    "family"
                ]
            ] += 1

            objective_counts[
                record[
                    "attack_objective"
                ]
            ] += 1

            template_counts[
                record[
                    "template_id"
                ]
            ] += 1

            position_counts[
                record[
                    "insertion_position"
                ]
            ] += 1

            representation_counts[
                record[
                    "representation"
                ]
            ] += 1

        distributions[
            track_name
        ] = {
            "family":
                dict(
                    sorted(
                        family_counts.items()
                    )
                ),

            "attack_objective":
                dict(
                    sorted(
                        objective_counts.items()
                    )
                ),

            "template_id":
                dict(
                    sorted(
                        template_counts.items()
                    )
                ),

            "insertion_position":
                dict(
                    sorted(
                        position_counts.items()
                    )
                ),

            "representation":
                dict(
                    sorted(
                        representation_counts.items()
                    )
                ),
        }

        if family_counts:
            spread = (
                max(
                    family_counts.values()
                )
                -
                min(
                    family_counts.values()
                )
            )

            if spread > 1:
                add_error(
                    errors,
                    "family_distribution_unbalanced",
                    {
                        "track":
                            track_name,

                        "spread":
                            spread,
                    },
                )

    for base_text_id, splits in (
        base_split_usage.items()
    ):
        if len(splits) > 1:
            add_error(
                errors,
                "cross_base_split_leakage",
                {
                    "base_text_id":
                        base_text_id,

                    "splits":
                        sorted(splits),
                },
            )

    development_tracks = (
        "train",
        "validation_seen",
        "validation_unseen",
    )

    ood_families = set(
        read_json(
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
        )[
            "ood_families"
        ]
    )

    for track_name in development_tracks:
        for record in track_records.get(
            track_name,
            [],
        ):
            if (
                record[
                    "family"
                ]
                in ood_families
            ):
                add_error(
                    errors,
                    "ood_family_in_development_track",
                    {
                        "track":
                            track_name,

                        "template_id":
                            record[
                                "template_id"
                            ],
                    },
                )

    shared_groups = (
        config[
            "render_plan"
        ][
            "shared_groups"
        ]
    )

    for group_name, group in (
        shared_groups.items()
    ):
        tracks = group[
            "tracks"
        ]

        if len(tracks) < 2:
            continue

        reference = {
            record[
                "base_text_id"
            ]:
                record[
                    "render_plan_id"
                ]

            for record in track_records.get(
                tracks[0],
                [],
            )
        }

        for track_name in tracks[1:]:
            current = {
                record[
                    "base_text_id"
                ]:
                    record[
                        "render_plan_id"
                    ]

                for record
                in track_records.get(
                    track_name,
                    [],
                )
            }

            if current != reference:
                add_error(
                    errors,
                    "shared_group_render_plan_mismatch",
                    {
                        "group":
                            group_name,

                        "track":
                            track_name,
                    },
                )

    error_summary = Counter(
        error[
            "reason"
        ]
        for error in errors
    )

    report = {
        "step":
            "3D.2",

        "status":
            "PASS"
            if not errors
            else "FAIL",

        "errors":
            len(errors),

        "render_plans":
            len(render_plans),

        "assignments":
            sum(
                len(records)
                for records
                in track_records.values()
            ),

        "generation_config_sha256":
            config_sha256,

        "distributions":
            distributions,

        "error_summary":
            dict(
                sorted(
                    error_summary.items()
                )
            ),

        "error_details":
            errors[:1000],
    }

    report_path = (
        output_root
        /
        "plan_validation_report.json"
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
        "FEVER Injection Plan Validation"
    )

    print(
        "================================"
    )

    print(
        "\nRender plans:",
        len(render_plans),
    )

    print(
        "Assignments:",
        report[
            "assignments"
        ],
    )

    print(
        "\nErrors:",
        len(errors),
    )

    if error_summary:
        print(
            "\nError summary:"
        )

        for reason, count in sorted(
            error_summary.items()
        ):
            print(
                f"  {reason}: {count}"
            )

    print(
        "Status:",
        report[
            "status"
        ],
    )

    print(
        "\nReport:",
        report_path,
    )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()