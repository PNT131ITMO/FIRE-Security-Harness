from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_fever_injected_contexts import (
    build_sample,
    load_base_contexts,
    load_registry,
    read_jsonl,
    read_yaml,
    sha256_file,
    validate_assignment_against_registry,
)


TRACK_ORDER = (
    "train",
    "validation_seen",
    "validation_unseen",
    "paper_test_seen",
    "paper_test_ood_template",
    "paper_test_ood_family",
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

    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    config = read_yaml(
        config_path
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

    assignments_dir = (
        output_root
        /
        "plans"
        /
        "assignments"
    )

    print(
        "Loading FEVER contexts..."
    )

    base_contexts = load_base_contexts(
        config
    )

    print(
        "Loading registry..."
    )

    templates, payloads = load_registry(
        config
    )

    report = {
        "tracks": {},
    }

    total_duplicate_groups = 0
    total_duplicate_records = 0

    for track_name in TRACK_ORDER:
        print(
            f"Checking {track_name}..."
        )

        track_config = config[
            "tracks"
        ][
            track_name
        ]

        base_split = track_config[
            "base_split"
        ]

        assignments = read_jsonl(
            assignments_dir
            /
            f"{track_name}.jsonl"
        )

        seen = {}
        duplicates = []

        classification = Counter()

        for assignment in assignments:
            base_text_id = assignment[
                "base_text_id"
            ]

            clean_record = (
                base_contexts[
                    base_split
                ][
                    base_text_id
                ]
            )

            template = templates[
                assignment[
                    "template_id"
                ]
            ]

            validate_assignment_against_registry(
                assignment,
                template,
                payloads,
            )

            sample = build_sample(
                clean_record,
                assignment,
                template,
                config,
                config_sha256,
            )

            injected_hash = sample[
                "injected_text_sha256"
            ]

            current = {
                "base_text_id":
                    base_text_id,

                "clean_text_sha256":
                    sample[
                        "clean_text_sha256"
                    ],

                "template_id":
                    sample[
                        "template_id"
                    ],

                "payload_ids":
                    sample[
                        "payload_ids"
                    ],

                "insertion_position":
                    sample[
                        "insertion_position"
                    ],

                "representation":
                    sample[
                        "representation"
                    ],

                "render_plan_id":
                    sample[
                        "render_plan_id"
                    ],

                "assignment_id":
                    sample[
                        "assignment_id"
                    ],

                "injected_text_sha256":
                    injected_hash,
            }

            previous = seen.get(
                injected_hash
            )

            if previous is None:
                seen[
                    injected_hash
                ] = current

                continue

            same_clean_text = (
                previous[
                    "clean_text_sha256"
                ]
                ==
                current[
                    "clean_text_sha256"
                ]
            )

            same_template = (
                previous[
                    "template_id"
                ]
                ==
                current[
                    "template_id"
                ]
            )

            same_payload = (
                previous[
                    "payload_ids"
                ]
                ==
                current[
                    "payload_ids"
                ]
            )

            same_position = (
                previous[
                    "insertion_position"
                ]
                ==
                current[
                    "insertion_position"
                ]
            )

            same_representation = (
                previous[
                    "representation"
                ]
                ==
                current[
                    "representation"
                ]
            )

            if (
                same_clean_text
                and
                same_template
                and
                same_payload
                and
                same_position
                and
                same_representation
            ):
                kind = (
                    "same_clean_same_attack"
                )

            elif same_clean_text:
                kind = (
                    "same_clean_different_attack_metadata"
                )

            else:
                kind = (
                    "different_clean_same_injected_text"
                )

            classification[
                kind
            ] += 1

            duplicates.append(
                {
                    "type":
                        kind,

                    "first":
                        previous,

                    "duplicate":
                        current,
                }
            )

        duplicate_groups = len(
            {
                item[
                    "duplicate"
                ][
                    "injected_text_sha256"
                ]
                for item in duplicates
            }
        )

        total_duplicate_groups += (
            duplicate_groups
        )

        total_duplicate_records += len(
            duplicates
        )

        report[
            "tracks"
        ][
            track_name
        ] = {
            "records":
                len(assignments),

            "unique_injected_texts":
                len(seen),

            "duplicate_groups":
                duplicate_groups,

            "duplicate_records":
                len(duplicates),

            "classification":
                dict(
                    sorted(
                        classification.items()
                    )
                ),

            "examples":
                duplicates[
                    :args.max_examples
                ],
        }

    report[
        "total_duplicate_groups"
    ] = total_duplicate_groups

    report[
        "total_duplicate_records"
    ] = total_duplicate_records

    report_path = (
        output_root
        /
        "duplicate_diagnostic_report.json"
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
        "Injected Text Duplicate Diagnostic"
    )

    print(
        "================================"
    )

    for track_name in TRACK_ORDER:
        info = report[
            "tracks"
        ][
            track_name
        ]

        print(
            f"\n{track_name}:"
        )

        print(
            "  records:",
            info[
                "records"
            ],
        )

        print(
            "  unique injected texts:",
            info[
                "unique_injected_texts"
            ],
        )

        print(
            "  duplicate groups:",
            info[
                "duplicate_groups"
            ],
        )

        print(
            "  duplicate records:",
            info[
                "duplicate_records"
            ],
        )

        for kind, count in (
            info[
                "classification"
            ].items()
        ):
            print(
                f"  {kind}: {count}"
            )

    print(
        "\nTotal duplicate groups:",
        total_duplicate_groups,
    )

    print(
        "Total duplicate records:",
        total_duplicate_records,
    )

    print(
        "\nReport:",
        report_path,
    )


if __name__ == "__main__":
    main()