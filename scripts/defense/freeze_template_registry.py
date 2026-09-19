from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


REGISTRY_FILES = (
    "family_registry.json",
    "payload_registry.jsonl",
    "train.jsonl",
    "validation.jsonl",
    "ood_template.jsonl",
    "ood_family.jsonl",
)

AUDIT_FILES = (
    "contamination_report.json",
    "contamination_pairs.jsonl",
    "validation_report.json",
    "validation_errors.jsonl",
)

SCRIPT_FILES = (
    "audit_template_registry_contamination.py",
    "validate_template_registry.py",
    "freeze_template_registry.py",
)

TEMPLATE_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "ood_template": "ood_template.jsonl",
    "ood_family": "ood_family.jsonl",
}


def sha256_file(
    path: Path,
) -> str:

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
            digest.update(
                chunk
            )

    return digest.hexdigest()


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
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"{path}:{line_number}: {error}"
                ) from error

    return records


def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


def file_entry(
    path: Path,
    records: int | None = None,
) -> dict:

    entry = {
        "path":
            str(
                path.as_posix()
            ),

        "sha256":
            sha256_file(
                path
            ),
    }

    if records is not None:

        entry[
            "records"
        ] = records

    return entry


def aggregate_registry_hash(
    artifacts: dict,
) -> str:

    digest = hashlib.sha256()

    for name in sorted(
        artifacts
    ):

        sha = artifacts[
            name
        ][
            "sha256"
        ]

        digest.update(
            (
                f"{name}\0{sha}\n"
            ).encode(
                "utf-8"
            )
        )

    return digest.hexdigest()


def collect_template_stats(
    registry_dir: Path,
) -> dict:

    split_counts = {}
    family_counts = {}
    cluster_counts = {}
    objective_counts = {}

    total_templates = 0
    all_clusters = set()

    for split, file_name in (
        TEMPLATE_FILES.items()
    ):

        path = (
            registry_dir
            /
            file_name
        )

        records = read_jsonl(
            path
        )

        split_counts[
            split
        ] = len(
            records
        )

        total_templates += len(
            records
        )

        families = Counter()
        objectives = Counter()
        clusters = set()

        for record in records:

            family = record.get(
                "family"
            )

            objective = record.get(
                "attack_objective"
            )

            cluster = record.get(
                "semantic_cluster_id"
            )

            if family is not None:

                families[
                    family
                ] += 1

            if objective is not None:

                objectives[
                    objective
                ] += 1

            if cluster is not None:

                clusters.add(
                    cluster
                )

                all_clusters.add(
                    cluster
                )

        family_counts[
            split
        ] = dict(
            sorted(
                families.items()
            )
        )

        objective_counts[
            split
        ] = dict(
            sorted(
                objectives.items()
            )
        )

        cluster_counts[
            split
        ] = len(
            clusters
        )

    return {
        "template_counts":
            split_counts,

        "total_templates":
            total_templates,

        "family_counts":
            family_counts,

        "objective_counts":
            objective_counts,

        "semantic_cluster_counts":
            cluster_counts,

        "total_semantic_clusters":
            len(
                all_clusters
            ),
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--registry-dir",
        default=(
            "data/processed/defense/"
            "template_registry"
        ),
    )

    parser.add_argument(
        "--scripts-dir",
        default=(
            "scripts/defense"
        ),
    )

    args = parser.parse_args()

    registry_dir = Path(
        args.registry_dir
    )

    scripts_dir = Path(
        args.scripts_dir
    )

    for file_name in REGISTRY_FILES:

        require_file(
            registry_dir
            /
            file_name
        )

    for file_name in AUDIT_FILES:

        require_file(
            registry_dir
            /
            file_name
        )

    for file_name in SCRIPT_FILES:

        require_file(
            scripts_dir
            /
            file_name
        )

    family_registry = read_json(
        registry_dir
        /
        "family_registry.json"
    )

    validation_report = read_json(
        registry_dir
        /
        "validation_report.json"
    )

    contamination_report = read_json(
        registry_dir
        /
        "contamination_report.json"
    )

    validation_errors = read_jsonl(
        registry_dir
        /
        "validation_errors.jsonl"
    )

    contamination_pairs = read_jsonl(
        registry_dir
        /
        "contamination_pairs.jsonl"
    )

    if (
        validation_report.get(
            "status"
        )
        != "PASS"
    ):

        raise RuntimeError(
            "Registry validation has not passed"
        )

    if (
        validation_report.get(
            "errors"
        )
        != 0
    ):

        raise RuntimeError(
            "Registry validation contains errors"
        )

    if validation_errors:

        raise RuntimeError(
            "validation_errors.jsonl is not empty"
        )

    if (
        contamination_report.get(
            "status"
        )
        != "PASS"
    ):

        raise RuntimeError(
            "Contamination audit has not passed"
        )

    if (
        contamination_report.get(
            "blocking_matches"
        )
        != 0
    ):

        raise RuntimeError(
            "Contamination audit contains blocking matches"
        )

    semantic_check = (
        contamination_report.get(
            "semantic_check",
            {},
        )
    )

    if (
        semantic_check.get(
            "ran"
        )
        is not True
    ):

        raise RuntimeError(
            "Semantic contamination audit was not run"
        )

    if not semantic_check.get(
        "model"
    ):

        raise RuntimeError(
            "Semantic contamination model is missing"
        )

    stats = collect_template_stats(
        registry_dir
    )

    payloads = read_jsonl(
        registry_dir
        /
        "payload_registry.jsonl"
    )

    artifacts = {}

    for file_name in REGISTRY_FILES:

        path = (
            registry_dir
            /
            file_name
        )

        records = None

        if (
            path.suffix
            == ".jsonl"
        ):

            records = len(
                read_jsonl(
                    path
                )
            )

        artifacts[
            file_name
        ] = file_entry(
            path,
            records,
        )

    audit_artifacts = {}

    for file_name in AUDIT_FILES:

        path = (
            registry_dir
            /
            file_name
        )

        records = None

        if (
            path.suffix
            == ".jsonl"
        ):

            records = len(
                read_jsonl(
                    path
                )
            )

        audit_artifacts[
            file_name
        ] = file_entry(
            path,
            records,
        )

    pipeline_scripts = {}

    for file_name in SCRIPT_FILES:

        path = (
            scripts_dir
            /
            file_name
        )

        pipeline_scripts[
            file_name
        ] = file_entry(
            path
        )

    registry_sha256 = (
        aggregate_registry_hash(
            artifacts
        )
    )

    manifest = {
        "artifact":
            "Attack Template Registry",

        "registry_version":
            family_registry.get(
                "registry_version",
                "1.0",
            ),

        "taxonomy_version":
            family_registry.get(
                "taxonomy_version",
                "1.0",
            ),

        "step":
            "3C",

        "status":
            "frozen",

        "frozen":
            True,

        "mutable":
            False,

        "registry_sha256":
            registry_sha256,

        "source_policy": {
            "template_source":
                "handcrafted",

            "llm_generated_templates":
                False,

            "fever_content_used":
                False,

            "fever_gold_labels_used":
                False,

            "bipia_used_as_template_source":
                False,

            "notinject_used_as_template_source":
                False,

            "paper_test_used_for_template_design":
                False,
        },

        "experimental_policy": {
            "train":
                "development_training",

            "validation":
                "development_model_selection",

            "ood_template":
                "evaluation_only",

            "ood_family":
                "evaluation_only",

            "ood_template_tuning_allowed":
                False,

            "ood_family_tuning_allowed":
                False,

            "semantic_clusters_cross_split_allowed":
                False,

            "multi_mechanism_templates_allowed":
                False,

            "ambiguous_family_templates_allowed":
                False,
        },

        "ood_policy": {
            "ood_template":
                {
                    "family_seen_in_development":
                        True,

                    "objective_seen_in_development":
                        True,

                    "semantic_cluster_seen_in_development":
                        False,
                },

            "ood_family":
                {
                    "family_seen_in_development_templates":
                        False,

                    "development_use_allowed":
                        False,

                    "full_training_data_absence_verified":
                        False,

                    "full_training_data_absence_verification_stage":
                        "dataset_assembly_or_final_training_audit",
                },
        },

        "normalization_policy": {
            "contamination_text":
                (
                    "Unicode NFKC + strip + "
                    "collapse whitespace + casefold"
                ),

            "template_hash":
                "SHA-256 of exact UTF-8 template string",

            "payload_hash":
                "SHA-256 of exact UTF-8 payload value",

            "registry_hash":
                (
                    "SHA-256 over sorted artifact "
                    "name and artifact SHA-256 pairs"
                ),
        },

        "counts": {
            **stats,

            "payloads":
                len(
                    payloads
                ),

            "contamination_pairs":
                len(
                    contamination_pairs
                ),
        },

        "development_families":
            family_registry.get(
                "development_families",
                [],
            ),

        "ood_families":
            family_registry.get(
                "ood_families",
                [],
            ),

        "objective_vocabulary":
            family_registry.get(
                "objective_vocabulary",
                [],
            ),

        "contamination_audit": {
            "status":
                contamination_report.get(
                    "status"
                ),

            "blocking_matches":
                contamination_report.get(
                    "blocking_matches"
                ),

            "matches":
                contamination_report.get(
                    "matches"
                ),

            "semantic_check":
                semantic_check,

            "reference_strings_audited":
                contamination_report.get(
                    "reference_strings_audited"
                ),

            "reference_groups":
                contamination_report.get(
                    "reference_groups",
                    {},
                ),

            "thresholds":
                contamination_report.get(
                    "thresholds",
                    {},
                ),
        },

        "strict_validation": {
            "status":
                validation_report.get(
                    "status"
                ),

            "errors":
                validation_report.get(
                    "errors"
                ),

            "warnings":
                validation_report.get(
                    "warnings"
                ),

            "total_templates":
                validation_report.get(
                    "total_templates"
                ),

            "payload_count":
                validation_report.get(
                    "payload_count"
                ),

            "total_semantic_clusters":
                validation_report.get(
                    "total_semantic_clusters"
                ),
        },

        "artifacts":
            artifacts,

        "audit_artifacts":
            audit_artifacts,

        "pipeline_scripts":
            pipeline_scripts,

        "freeze_policy": {
            "edit_after_freeze_allowed":
                False,

            "required_action_for_change":
                "create_new_registry_version",

            "rerun_contamination_audit_after_change":
                True,

            "rerun_strict_validation_after_change":
                True,

            "rerun_freeze_after_change":
                True,
        },

        "next_stage":
            "3D_FEVER_derived_injection_generation",
    }

    manifest_path = (
        registry_dir
        /
        "manifest.json"
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
        "\n================================"
    )

    print(
        "Template Registry Freeze"
    )

    print(
        "================================"
    )

    print(
        "\nRegistry version:",
        manifest[
            "registry_version"
        ],
    )

    print(
        "Taxonomy version:",
        manifest[
            "taxonomy_version"
        ],
    )

    print(
        "\nTemplates:",
        stats[
            "total_templates"
        ],
    )

    print(
        "Payloads:",
        len(
            payloads
        ),
    )

    print(
        "Semantic clusters:",
        stats[
            "total_semantic_clusters"
        ],
    )

    print(
        "\nValidation:",
        validation_report.get(
            "status"
        ),
    )

    print(
        "Contamination:",
        contamination_report.get(
            "status"
        ),
    )

    print(
        "Blocking matches:",
        contamination_report.get(
            "blocking_matches"
        ),
    )

    print(
        "Semantic audit:",
        semantic_check.get(
            "ran"
        ),
    )

    print(
        "\nFrozen:",
        manifest[
            "frozen"
        ],
    )

    print(
        "Registry SHA-256:",
        registry_sha256,
    )

    print(
        "\nManifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()