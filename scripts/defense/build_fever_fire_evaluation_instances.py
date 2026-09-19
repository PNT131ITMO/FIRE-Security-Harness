from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml


EVAL_TRACKS = (
    "validation_seen",
    "validation_unseen",
    "paper_test_seen",
    "paper_test_ood_template",
    "paper_test_ood_family",
)

FEVER_TO_FIRE = {
    "SUPPORTS": True,
    "REFUTES": False,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def stable_id(
    prefix: str,
    components: list,
) -> str:
    payload = json.dumps(
        components,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        f"{prefix}_"
        f"{sha256_text(payload)}"
    )


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


def write_jsonl(
    path: Path,
    records: list[dict],
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
                    separators=(",", ":"),
                )
                + "\n"
            )


def canonical_claim_id(
    value,
) -> str:
    if value is None:
        raise RuntimeError(
            "Missing claim ID"
        )

    return str(value)


def get_claim_id(
    record: dict,
):
    value = record.get(
        "claim_id"
    )

    if value is None:
        value = record.get(
            "id"
        )

    return value


def get_claim_text(
    record: dict,
) -> str:
    claim = record.get(
        "claim"
    )

    if (
        not isinstance(claim, str)
        or not claim.strip()
    ):
        raise RuntimeError(
            "Missing or invalid FEVER claim"
        )

    return claim


def parse_bool_label(
    value,
) -> bool | None:
    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        if value == 1:
            return True

        if value == 0:
            return False

    if isinstance(
        value,
        str,
    ):
        normalized = (
            value.strip()
            .casefold()
        )

        if normalized in {
            "true",
            "1",
        }:
            return True

        if normalized in {
            "false",
            "0",
        }:
            return False

    return None


def get_fever_label(
    record: dict,
) -> str:
    fever_label = record.get(
        "fever_label"
    )

    if isinstance(
        fever_label,
        str,
    ):
        normalized = (
            fever_label.strip()
            .upper()
        )

        if (
            normalized
            in FEVER_TO_FIRE
        ):
            return normalized

    label = record.get(
        "label"
    )

    if isinstance(
        label,
        str,
    ):
        normalized = (
            label.strip()
            .upper()
        )

        if (
            normalized
            in FEVER_TO_FIRE
        ):
            return normalized

    fire_label = record.get(
        "fire_label"
    )

    parsed_fire = parse_bool_label(
        fire_label
    )

    if parsed_fire is None:
        parsed_fire = parse_bool_label(
            label
        )

    if parsed_fire is True:
        return "SUPPORTS"

    if parsed_fire is False:
        return "REFUTES"

    raise RuntimeError(
        "Cannot determine FEVER label"
    )


def get_fire_label(
    record: dict,
    fever_label: str,
) -> bool:
    fire_label = parse_bool_label(
        record.get(
            "fire_label"
        )
    )

    if fire_label is not None:
        expected = (
            FEVER_TO_FIRE[
                fever_label
            ]
        )

        if (
            fire_label
            != expected
        ):
            raise RuntimeError(
                "FEVER/FIRE label conflict"
            )

        return fire_label

    raw_label = record.get(
        "label"
    )

    parsed_label = parse_bool_label(
        raw_label
    )

    if parsed_label is not None:
        expected = (
            FEVER_TO_FIRE[
                fever_label
            ]
        )

        if (
            parsed_label
            != expected
        ):
            raise RuntimeError(
                "FEVER/FIRE label conflict"
            )

        return parsed_label

    return FEVER_TO_FIRE[
        fever_label
    ]


def load_gold_index(
    config: dict,
) -> dict[
    str,
    dict[str, dict],
]:
    result = {}

    split_paths = (
        config[
            "inputs"
        ][
            "fever_gold"
        ][
            "splits"
        ]
    )

    for split, path_string in (
        split_paths.items()
    ):
        path = Path(
            path_string
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        records = read_jsonl(
            path
        )

        index = {}

        for line_number, record in enumerate(
            records,
            start=1,
        ):
            claim_id = get_claim_id(
                record
            )

            try:
                claim_key = (
                    canonical_claim_id(
                        claim_id
                    )
                )

                claim = get_claim_text(
                    record
                )

                fever_label = (
                    get_fever_label(
                        record
                    )
                )

                fire_label = (
                    get_fire_label(
                        record,
                        fever_label,
                    )
                )

            except Exception as error:
                raise RuntimeError(
                    f"{path}:{line_number}: "
                    f"{error}"
                ) from error

            if claim_key in index:
                raise RuntimeError(
                    f"{split}: duplicate claim_id "
                    f"{claim_key}"
                )

            index[
                claim_key
            ] = {
                "claim_id":
                    claim_id,

                "claim_key":
                    claim_key,

                "claim":
                    claim,

                "gold_fever_label":
                    fever_label,

                "gold_fire_label":
                    fire_label,
            }

        result[
            split
        ] = index

    return result


def restore_clean_text(
    context_record: dict,
) -> str:
    injected_text = context_record.get(
        "text"
    )

    if not isinstance(
        injected_text,
        str,
    ):
        raise RuntimeError(
            "Invalid injected text"
        )

    start = context_record.get(
        "attack_start_char"
    )

    end = context_record.get(
        "attack_end_char"
    )

    if (
        not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise RuntimeError(
            "Invalid attack span"
        )

    if (
        start < 0
        or end < start
        or end > len(
            injected_text
        )
    ):
        raise RuntimeError(
            "Attack span out of range"
        )

    inserted_block = (
        context_record.get(
            "inserted_block"
        )
    )

    if not isinstance(
        inserted_block,
        str,
    ):
        raise RuntimeError(
            "Missing inserted_block"
        )

    actual_block = (
        injected_text[
            start:end
        ]
    )

    if (
        actual_block
        != inserted_block
    ):
        raise RuntimeError(
            "Attack span does not match "
            "inserted_block"
        )

    expected_block_sha = (
        context_record.get(
            "inserted_block_sha256"
        )
    )

    if (
        expected_block_sha
        != sha256_text(
            inserted_block
        )
    ):
        raise RuntimeError(
            "inserted_block_sha256 mismatch"
        )

    expected_injected_sha = (
        context_record.get(
            "injected_text_sha256"
        )
    )

    if (
        expected_injected_sha
        != sha256_text(
            injected_text
        )
    ):
        raise RuntimeError(
            "injected_text_sha256 mismatch"
        )

    clean_text = (
        injected_text[
            :start
        ]
        +
        injected_text[
            end:
        ]
    )

    expected_clean_sha = (
        context_record.get(
            "clean_text_sha256"
        )
    )

    if (
        expected_clean_sha
        != sha256_text(
            clean_text
        )
    ):
        raise RuntimeError(
            "clean_text_sha256 mismatch"
        )

    return clean_text


def attack_target(
    objective: str,
) -> bool | None:
    if objective == "force_true":
        return True

    if objective == "force_false":
        return False

    return None


def target_relation(
    target: bool | None,
    gold: bool,
) -> str:
    if target is None:
        return "not_applicable"

    if target == gold:
        return "same_as_gold"

    return "opposite_of_gold"


def targeted_success_criterion(
    target: bool | None,
) -> str | None:
    if target is True:
        return "fire_answer_equals_true"

    if target is False:
        return "fire_answer_equals_false"

    return None


def validate_context_claim_provenance(
    context_record: dict,
    gold: dict,
) -> None:
    fever_labels = (
        context_record.get(
            "fever_labels"
        )
    )

    if not isinstance(
        fever_labels,
        list,
    ):
        raise RuntimeError(
            "Invalid fever_labels"
        )

    normalized_labels = {
        str(label)
        .strip()
        .upper()
        for label in fever_labels
    }

    if (
        gold[
            "gold_fever_label"
        ]
        not in normalized_labels
    ):
        raise RuntimeError(
            "Claim gold label is not present "
            "in context fever_labels"
        )


def build_evaluation_instance(
    track_name: str,
    track_config: dict,
    context_record: dict,
    gold: dict,
    clean_text: str,
    config_sha256: str,
) -> dict:
    objective = context_record[
        "attack_objective"
    ]

    target = attack_target(
        objective
    )

    relation = target_relation(
        target,
        gold[
            "gold_fire_label"
        ],
    )

    targeted_eligible = (
        target is not None
        and
        relation
        == "opposite_of_gold"
    )

    criterion = (
        targeted_success_criterion(
            target
        )
        if targeted_eligible
        else None
    )

    evaluation_instance_id = (
        stable_id(
            "fever_fire_eval",
            [
                track_name,
                context_record[
                    "sample_id"
                ],
                gold[
                    "claim_key"
                ],
                context_record[
                    "registry_sha256"
                ],
                context_record[
                    "generator_version"
                ],
                config_sha256,
            ],
        )
    )

    final_evaluation = (
        track_config[
            "final_evaluation"
        ]
    )

    return {
        "evaluation_instance_id":
            evaluation_instance_id,

        "dataset_split":
            track_name,

        "evaluation_role":
            track_config[
                "role"
            ],

        "final_evaluation":
            final_evaluation,

        "locked":
            final_evaluation,

        "allow_training":
            track_config[
                "allow_training"
            ],

        "allow_model_selection":
            track_config[
                "allow_model_selection"
            ],

        "allow_threshold_tuning":
            track_config[
                "allow_threshold_tuning"
            ],

        "injected_context_id":
            context_record[
                "sample_id"
            ],

        "base_text_id":
            context_record[
                "base_text_id"
            ],

        "render_plan_id":
            context_record[
                "render_plan_id"
            ],

        "claim_id":
            gold[
                "claim_id"
            ],

        "claim":
            gold[
                "claim"
            ],

        "gold_fever_label":
            gold[
                "gold_fever_label"
            ],

        "gold_fire_label":
            gold[
                "gold_fire_label"
            ],

        "label":
            gold[
                "gold_fire_label"
            ],

        "claim_ids_in_context":
            context_record[
                "claim_ids"
            ],

        "fever_labels_in_context":
            context_record[
                "fever_labels"
            ],

        "evidence_refs":
            context_record[
                "evidence_refs"
            ],

        "context_size":
            context_record[
                "context_size"
            ],

        "clean_text_sha256":
            sha256_text(
                clean_text
            ),

        "injected_text_sha256":
            context_record[
                "injected_text_sha256"
            ],

        "attack_text_sha256":
            context_record[
                "attack_text_sha256"
            ],

        "inserted_block_sha256":
            context_record[
                "inserted_block_sha256"
            ],

        "attack_start_char":
            context_record[
                "attack_start_char"
            ],

        "attack_end_char":
            context_record[
                "attack_end_char"
            ],

        "clean_insertion_offset":
            context_record[
                "clean_insertion_offset"
            ],

        "template_id":
            context_record[
                "template_id"
            ],

        "template_split":
            context_record[
                "template_split"
            ],

        "family":
            context_record[
                "family"
            ],

        "attack_objective":
            objective,

        "semantic_cluster_id":
            context_record[
                "semantic_cluster_id"
            ],

        "payload_ids":
            context_record[
                "payload_ids"
            ],

        "insertion_position":
            context_record[
                "insertion_position"
            ],

        "representation":
            context_record[
                "representation"
            ],

        "encoding":
            context_record[
                "encoding"
            ],

        "attack_target":
            target,

        "attack_target_relation":
            relation,

        "general_fire_evaluable":
            True,

        "targeted_asr_evaluable":
            targeted_eligible,

        "success_criterion":
            criterion,

        "source":
            "fever_derived",

        "original_source":
            "FEVER",

        "synthetic":
            True,

        "upstream_split":
            context_record[
                "upstream_split"
            ],

        "registry_version":
            context_record[
                "registry_version"
            ],

        "registry_sha256":
            context_record[
                "registry_sha256"
            ],

        "generator_version":
            context_record[
                "generator_version"
            ],

        "generation_seed":
            context_record[
                "generation_seed"
            ],

        "generation_config_sha256":
            config_sha256,
    }


def distribution(
    records: list[dict],
    field: str,
) -> dict:
    counter = Counter()

    for record in records:
        value = record.get(
            field
        )

        if isinstance(
            value,
            bool,
        ):
            key = str(
                value
            )

        elif value is None:
            key = "None"

        else:
            key = str(
                value
            )

        counter[
            key
        ] += 1

    return dict(
        sorted(
            counter.items()
        )
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

    spec_manifest_path = (
        output_root
        /
        "spec_manifest.json"
    )

    plan_validation_path = (
        output_root
        /
        "plan_validation_report.json"
    )

    generation_report_path = Path(
        config[
            "output"
        ][
            "generation_report"
        ]
    )

    if not spec_manifest_path.exists():
        raise RuntimeError(
            "Step 3D.1 manifest missing"
        )

    if not plan_validation_path.exists():
        raise RuntimeError(
            "Step 3D.2 validation missing"
        )

    if not generation_report_path.exists():
        raise RuntimeError(
            "Step 3D.3 generation report missing"
        )

    spec_manifest = read_json(
        spec_manifest_path
    )

    if (
        spec_manifest.get(
            "frozen"
        )
        is not True
    ):
        raise RuntimeError(
            "Step 3D.1 specification "
            "is not frozen"
        )

    if (
        spec_manifest[
            "generation_config"
        ][
            "sha256"
        ]
        != config_sha256
    ):
        raise RuntimeError(
            "Generation configuration "
            "changed after freeze"
        )

    plan_validation = read_json(
        plan_validation_path
    )

    if (
        plan_validation.get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Step 3D.2 validation "
            "has not passed"
        )

    generation_report = read_json(
        generation_report_path
    )

    if (
        generation_report.get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Step 3D.3 generation "
            "has not passed"
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
        generation_report.get(
            "registry_sha256"
        )
        != expected_registry_sha
    ):
        raise RuntimeError(
            "Generation report registry "
            "SHA-256 mismatch"
        )

    print(
        "Loading FEVER gold claims..."
    )

    gold_index = load_gold_index(
        config
    )

    fire_eval_root = Path(
        config[
            "output"
        ][
            "fire_eval_dir"
        ]
    )

    targeted_root = (
        fire_eval_root
        /
        "targeted_asr"
    )

    all_instance_ids = set()

    output_files = {}
    targeted_files = {}
    track_reports = {}

    total_instances = 0
    total_targeted = 0

    print(
        "Building FIRE evaluation instances..."
    )

    for track_name in EVAL_TRACKS:
        track_config = (
            config[
                "tracks"
            ][
                track_name
            ]
        )

        if (
            track_config[
                "allow_training"
            ]
            is not False
        ):
            raise RuntimeError(
                f"{track_name}: "
                "FIRE evaluation track "
                "cannot allow training"
            )

        base_split = (
            track_config[
                "base_split"
            ]
        )

        context_path = (
            output_root
            /
            track_config[
                "output"
            ]
        )

        if not context_path.exists():
            raise FileNotFoundError(
                context_path
            )

        contexts = read_jsonl(
            context_path
        )

        expected_contexts = (
            track_config[
                "expected_records"
            ]
        )

        if (
            len(contexts)
            != expected_contexts
        ):
            raise RuntimeError(
                f"{track_name}: expected "
                f"{expected_contexts} contexts, "
                f"got {len(contexts)}"
            )

        instances = []
        targeted_instances = []

        seen_pairs = set()

        claim_multiplicity = Counter()

        for context_record in contexts:
            if (
                context_record.get(
                    "dataset_split"
                )
                != track_name
            ):
                raise RuntimeError(
                    f"{track_name}: "
                    "dataset_split mismatch"
                )

            if (
                context_record.get(
                    "upstream_split"
                )
                != base_split
            ):
                raise RuntimeError(
                    f"{track_name}: "
                    "upstream_split mismatch"
                )

            if (
                context_record.get(
                    "registry_sha256"
                )
                != expected_registry_sha
            ):
                raise RuntimeError(
                    f"{track_name}: "
                    "registry SHA mismatch"
                )

            clean_text = (
                restore_clean_text(
                    context_record
                )
            )

            claim_ids = (
                context_record.get(
                    "claim_ids"
                )
            )

            if (
                not isinstance(
                    claim_ids,
                    list,
                )
                or not claim_ids
            ):
                raise RuntimeError(
                    f"{track_name}: "
                    "context has no claim_ids"
                )

            claim_keys = [
                canonical_claim_id(
                    claim_id
                )
                for claim_id
                in claim_ids
            ]

            if (
                len(claim_keys)
                != len(
                    set(
                        claim_keys
                    )
                )
            ):
                raise RuntimeError(
                    f"{track_name}: duplicate "
                    "claim_id inside context"
                )

            claim_multiplicity[
                len(
                    claim_keys
                )
            ] += 1

            for claim_key in claim_keys:
                gold = (
                    gold_index[
                        base_split
                    ].get(
                        claim_key
                    )
                )

                if gold is None:
                    raise RuntimeError(
                        f"{track_name}: "
                        f"claim_id={claim_key} "
                        f"not found in "
                        f"FEVER {base_split}"
                    )

                validate_context_claim_provenance(
                    context_record,
                    gold,
                )

                pair_key = (
                    context_record[
                        "sample_id"
                    ],
                    claim_key,
                )

                if pair_key in seen_pairs:
                    raise RuntimeError(
                        f"{track_name}: duplicate "
                        "context/claim pair"
                    )

                seen_pairs.add(
                    pair_key
                )

                instance = (
                    build_evaluation_instance(
                        track_name,
                        track_config,
                        context_record,
                        gold,
                        clean_text,
                        config_sha256,
                    )
                )

                instance_id = (
                    instance[
                        "evaluation_instance_id"
                    ]
                )

                if (
                    instance_id
                    in all_instance_ids
                ):
                    raise RuntimeError(
                        "Duplicate "
                        "evaluation_instance_id: "
                        f"{instance_id}"
                    )

                all_instance_ids.add(
                    instance_id
                )

                instances.append(
                    instance
                )

                if (
                    instance[
                        "targeted_asr_evaluable"
                    ]
                    is True
                ):
                    targeted_instances.append(
                        instance
                    )

        instances.sort(
            key=lambda record: (
                str(
                    record[
                        "base_text_id"
                    ]
                ),
                str(
                    record[
                        "claim_id"
                    ]
                ),
                record[
                    "evaluation_instance_id"
                ],
            )
        )

        targeted_instances.sort(
            key=lambda record: (
                str(
                    record[
                        "base_text_id"
                    ]
                ),
                str(
                    record[
                        "claim_id"
                    ]
                ),
                record[
                    "evaluation_instance_id"
                ],
            )
        )

        output_path = (
            fire_eval_root
            /
            f"{track_name}.jsonl"
        )

        targeted_path = (
            targeted_root
            /
            f"{track_name}.jsonl"
        )

        write_jsonl(
            output_path,
            instances,
        )

        write_jsonl(
            targeted_path,
            targeted_instances,
        )

        output_files[
            track_name
        ] = {
            "path":
                str(
                    output_path.as_posix()
                ),

            "records":
                len(
                    instances
                ),

            "sha256":
                sha256_file(
                    output_path
                ),
        }

        targeted_files[
            track_name
        ] = {
            "path":
                str(
                    targeted_path.as_posix()
                ),

            "records":
                len(
                    targeted_instances
                ),

            "sha256":
                sha256_file(
                    targeted_path
                ),
        }

        track_reports[
            track_name
        ] = {
            "contexts":
                len(
                    contexts
                ),

            "claim_instances":
                len(
                    instances
                ),

            "targeted_asr_instances":
                len(
                    targeted_instances
                ),

            "claim_multiplicity":
                dict(
                    sorted(
                        (
                            str(key),
                            value,
                        )
                        for key, value
                        in claim_multiplicity.items()
                    )
                ),

            "gold_fever_label":
                distribution(
                    instances,
                    "gold_fever_label",
                ),

            "gold_fire_label":
                distribution(
                    instances,
                    "gold_fire_label",
                ),

            "family":
                distribution(
                    instances,
                    "family",
                ),

            "attack_objective":
                distribution(
                    instances,
                    "attack_objective",
                ),

            "attack_target_relation":
                distribution(
                    instances,
                    "attack_target_relation",
                ),

            "targeted_asr_evaluable":
                distribution(
                    instances,
                    "targeted_asr_evaluable",
                ),
        }

        total_instances += len(
            instances
        )

        total_targeted += len(
            targeted_instances
        )

    report = {
        "step":
            "3D.4",

        "status":
            "PASS",

        "policy": {
            "granularity":
                "claim",

            "context_expansion":
                "one_instance_per_claim_id",

            "arbitrary_claim_selection":
                False,

            "train_track_included":
                False,

            "general_fire_evaluation":
                "all_claim_instances",

            "targeted_asr":
                (
                    "force_true_or_force_false_"
                    "with_target_opposite_gold_only"
                ),

            "gold_label_used_for_attack_generation":
                False,

            "gold_label_used_for_evaluation_selection":
                True,

            "paper_test_training":
                False,

            "paper_test_model_selection":
                False,
        },

        "generation_config_sha256":
            config_sha256,

        "registry_sha256":
            expected_registry_sha,

        "total_claim_instances":
            total_instances,

        "total_targeted_asr_instances":
            total_targeted,

        "tracks":
            track_reports,

        "output_files":
            output_files,

        "targeted_asr_files":
            targeted_files,

        "next_stage":
            "3D.5_strict_validation_and_reproducibility",
    }

    report_path = (
        output_root
        /
        "fire_eval_report.json"
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
        "FEVER FIRE Evaluation Instances"
    )

    print(
        "================================"
    )

    for track_name in EVAL_TRACKS:
        track = track_reports[
            track_name
        ]

        print(
            f"\n{track_name}:"
        )

        print(
            "  contexts:",
            track[
                "contexts"
            ],
        )

        print(
            "  claim instances:",
            track[
                "claim_instances"
            ],
        )

        print(
            "  targeted ASR:",
            track[
                "targeted_asr_instances"
            ],
        )

    print(
        "\nTotal claim instances:",
        total_instances,
    )

    print(
        "Total targeted ASR instances:",
        total_targeted,
    )

    print(
        "\nStatus: PASS"
    )

    print(
        "Report:",
        report_path,
    )

    print(
        "Output:",
        fire_eval_root,
    )


if __name__ == "__main__":
    main()