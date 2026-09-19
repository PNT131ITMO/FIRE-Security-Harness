from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
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

UNRESOLVED_PLACEHOLDER = re.compile(
    r"\{\{[^{}]+\}\}"
)

BASE_EXCLUDED_FIELDS = {
    "sample_id",
    "text",
    "text_sha256",
    "label",
    "source",
    "original_source",
    "synthetic",
    "split",
    "dataset_split",
    "template_id",
    "template_split",
    "family",
    "attack_family",
    "attack_objective",
    "semantic_cluster_id",
    "payload_ids",
    "payload_bindings",
    "attack_text",
    "attack_text_sha256",
    "inserted_block",
    "inserted_block_sha256",
    "injected_text_sha256",
    "requested_position",
    "insertion_position",
    "position_adjusted",
    "representation",
    "encoding",
    "attack_start_char",
    "attack_end_char",
    "clean_insertion_offset",
    "render_plan_id",
    "assignment_id",
    "registry_version",
    "registry_sha256",
    "generator_version",
    "generation_seed",
    "generation_config_sha256",
    "detector_evaluable",
    "end_to_end_evaluable",
    "targeted_asr_evaluable",
    "success_criterion",
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
    raw = json.dumps(
        components,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        f"{prefix}_"
        f"{sha256_text(raw)}"
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


def load_base_contexts(
    config: dict,
) -> dict[str, dict[str, dict]]:
    result = {}

    for split, info in (
        config[
            "inputs"
        ][
            "fever_benign"
        ][
            "splits"
        ].items()
    ):
        records = read_jsonl(
            Path(
                info[
                    "path"
                ]
            )
        )

        if (
            len(records)
            != info[
                "expected_records"
            ]
        ):
            raise RuntimeError(
                f"{split}: context count mismatch"
            )

        index = {}

        for record in records:
            base_text_id = record.get(
                "base_text_id"
            )

            if not base_text_id:
                raise RuntimeError(
                    f"{split}: missing base_text_id"
                )

            if base_text_id in index:
                raise RuntimeError(
                    f"{split}: duplicate base_text_id "
                    f"{base_text_id}"
                )

            text = record.get(
                "text"
            )

            if (
                not isinstance(text, str)
                or not text
            ):
                raise RuntimeError(
                    f"{split}:{base_text_id}: "
                    "invalid clean text"
                )

            if not record.get(
                "base_text_sha256"
            ):
                raise RuntimeError(
                    f"{split}:{base_text_id}: "
                    "missing base_text_sha256"
                )

            for field in (
                "claim_ids",
                "fever_labels",
                "evidence_refs",
            ):
                if not isinstance(
                    record.get(field),
                    list,
                ):
                    raise RuntimeError(
                        f"{split}:{base_text_id}: "
                        f"invalid {field}"
                    )

            actual_clean_sha = (
                sha256_text(
                    text
                )
            )

            stored_text_sha = (
                record.get(
                    "text_sha256"
                )
            )

            if (
                stored_text_sha is not None
                and
                stored_text_sha
                != actual_clean_sha
            ):
                raise RuntimeError(
                    f"{split}:{base_text_id}: "
                    "Step 3B text_sha256 mismatch"
                )

            index[
                base_text_id
            ] = record

        result[
            split
        ] = index

    return result


def load_registry(
    config: dict,
) -> tuple[
    dict[str, dict],
    dict[str, dict],
]:
    registry = config[
        "inputs"
    ][
        "attack_registry"
    ]

    templates = {}

    for template_split, path_string in (
        registry[
            "template_splits"
        ].items()
    ):
        for record in read_jsonl(
            Path(path_string)
        ):
            template_id = record[
                "template_id"
            ]

            if template_id in templates:
                raise RuntimeError(
                    "Duplicate registry template_id: "
                    f"{template_id}"
                )

            templates[
                template_id
            ] = {
                **record,
                "_template_split":
                    template_split,
            }

    payloads = {}

    for record in read_jsonl(
        Path(
            registry[
                "payload_registry"
            ]
        )
    ):
        payload_id = record[
            "payload_id"
        ]

        if payload_id in payloads:
            raise RuntimeError(
                "Duplicate payload_id: "
                f"{payload_id}"
            )

        payloads[
            payload_id
        ] = record

    return templates, payloads


def validate_assignment_against_registry(
    assignment: dict,
    template: dict,
    payloads: dict[str, dict],
) -> None:
    if (
        template[
            "_template_split"
        ]
        != assignment[
            "template_split"
        ]
    ):
        raise RuntimeError(
            f"{assignment['assignment_id']}: "
            "template split mismatch"
        )

    for field in (
        "family",
        "attack_objective",
        "semantic_cluster_id",
    ):
        if (
            assignment[
                field
            ]
            != template[
                field
            ]
        ):
            raise RuntimeError(
                f"{assignment['assignment_id']}: "
                f"{field} mismatch"
            )

    if (
        assignment[
            "insertion_position"
        ]
        not in template[
            "allowed_positions"
        ]
    ):
        raise RuntimeError(
            f"{assignment['assignment_id']}: "
            "position not allowed"
        )

    if (
        assignment[
            "representation"
        ]
        not in template[
            "allowed_representations"
        ]
    ):
        raise RuntimeError(
            f"{assignment['assignment_id']}: "
            "representation not allowed"
        )

    if (
        assignment[
            "encoding"
        ]
        not in template[
            "allowed_encodings"
        ]
    ):
        raise RuntimeError(
            f"{assignment['assignment_id']}: "
            "encoding not allowed"
        )

    placeholders = (
        template.get(
            "placeholders",
            []
        )
    )

    bindings = assignment.get(
        "payload_bindings",
        {},
    )

    if (
        set(bindings)
        != set(placeholders)
    ):
        raise RuntimeError(
            f"{assignment['assignment_id']}: "
            "placeholder binding mismatch"
        )

    expected_payload_ids = []

    for placeholder in placeholders:
        binding = bindings[
            placeholder
        ]

        payload_id = binding[
            "payload_id"
        ]

        payload = payloads.get(
            payload_id
        )

        if payload is None:
            raise RuntimeError(
                f"{assignment['assignment_id']}: "
                f"unknown payload {payload_id}"
            )

        if (
            payload[
                "placeholder"
            ]
            != placeholder
        ):
            raise RuntimeError(
                f"{assignment['assignment_id']}: "
                "payload placeholder mismatch"
            )

        if (
            assignment[
                "attack_objective"
            ]
            not in payload[
                "compatible_objectives"
            ]
        ):
            raise RuntimeError(
                f"{assignment['assignment_id']}: "
                "payload objective mismatch"
            )

        if (
            binding[
                "value"
            ]
            != payload[
                "value"
            ]
        ):
            raise RuntimeError(
                f"{assignment['assignment_id']}: "
                "payload value mismatch"
            )

        expected_payload_ids.append(
            payload_id
        )

    if (
        assignment[
            "payload_ids"
        ]
        != expected_payload_ids
    ):
        raise RuntimeError(
            f"{assignment['assignment_id']}: "
            "payload_ids mismatch"
        )


def build_attack_text(
    template: dict,
    assignment: dict,
) -> str:
    text = template[
        "template"
    ]

    for placeholder in template.get(
        "placeholders",
        []
    ):
        value = (
            assignment[
                "payload_bindings"
            ][
                placeholder
            ][
                "value"
            ]
        )

        text = text.replace(
            "{{"
            + placeholder
            + "}}",
            value,
        )

    if UNRESOLVED_PLACEHOLDER.search(
        text
    ):
        raise RuntimeError(
            f"{assignment['assignment_id']}: "
            "unresolved placeholder"
        )

    return text


def apply_encoding(
    text: str,
    encoding: str,
) -> str:
    if encoding == "plain":
        return text

    raise RuntimeError(
        f"Unsupported encoding: {encoding}"
    )


def apply_representation(
    text: str,
    representation: str,
) -> str:
    if representation == "plain_text":
        return text

    if representation == "markdown":
        lines = text.split("\n")

        return "\n".join(
            (
                f"> {line}"
                if line
                else ">"
            )
            for line in lines
        )

    raise RuntimeError(
        "Unsupported representation: "
        f"{representation}"
    )


def separator_value(
    value: str,
) -> str:
    mapping = {
        "": "",
        "none": "",
        "double_lf": "\n\n",
        "single_lf": "\n",
    }

    if value not in mapping:
        raise RuntimeError(
            f"Unsupported separator: {value}"
        )

    return mapping[
        value
    ]


def extract_sentence_texts(
    record: dict,
) -> list[str] | None:
    simple_keys = (
        "sentence_texts",
        "context_sentences",
        "evidence_texts",
    )

    for key in simple_keys:
        value = record.get(
            key
        )

        if (
            isinstance(value, list)
            and value
            and all(
                isinstance(item, str)
                for item in value
            )
        ):
            return value

    structured_keys = (
        "sentences",
        "evidence_sentences",
        "evidence_items",
        "evidence",
    )

    for key in structured_keys:
        value = record.get(
            key
        )

        if (
            not isinstance(value, list)
            or not value
        ):
            continue

        texts = []

        for item in value:
            if not isinstance(
                item,
                dict,
            ):
                texts = []
                break

            text = (
                item.get("text")
                or item.get("sentence_text")
                or item.get("evidence_text")
            )

            if not isinstance(
                text,
                str,
            ):
                texts = []
                break

            texts.append(
                text
            )

        if texts:
            return texts

    return None


def offsets_from_sentence_texts(
    clean_text: str,
    sentence_texts: list[str],
    context_size: int,
) -> list[int] | None:
    if (
        len(sentence_texts)
        != context_size
    ):
        return None

    starts = []
    cursor = 0

    for sentence in sentence_texts:
        position = clean_text.find(
            sentence,
            cursor,
        )

        if position < 0:
            return None

        starts.append(
            position
        )

        cursor = (
            position
            + len(sentence)
        )

    if len(starts) < 2:
        return []

    return starts[1:]


def offsets_from_line_boundaries(
    clean_text: str,
    context_size: int,
) -> list[int] | None:
    required = (
        context_size
        - 1
    )

    if required <= 0:
        return []

    multi_newline = [
        match.end()
        for match in re.finditer(
            r"\n(?:[ \t]*\n)+",
            clean_text,
        )
    ]

    if (
        len(multi_newline)
        >= required
    ):
        return multi_newline[
            :required
        ]

    single_newline = [
        match.end()
        for match in re.finditer(
            r"\n",
            clean_text,
        )
    ]

    if (
        len(single_newline)
        >= required
    ):
        return single_newline[
            :required
        ]

    return None


def resolve_middle_offset(
    clean_record: dict,
) -> tuple[int, int, int, str]:
    clean_text = clean_record[
        "text"
    ]

    context_size = clean_record[
        "context_size"
    ]

    if context_size <= 1:
        raise RuntimeError(
            f"{clean_record['base_text_id']}: "
            "middle requested for singleton context"
        )

    sentence_texts = (
        extract_sentence_texts(
            clean_record
        )
    )

    offsets = None
    method = None

    if sentence_texts is not None:
        offsets = (
            offsets_from_sentence_texts(
                clean_text,
                sentence_texts,
                context_size,
            )
        )

        if offsets is not None:
            method = "sentence_text_alignment"

    if offsets is None:
        offsets = (
            offsets_from_line_boundaries(
                clean_text,
                context_size,
            )
        )

        if offsets is not None:
            method = "line_boundary"

    if (
        offsets is None
        or not offsets
    ):
        raise RuntimeError(
            f"{clean_record['base_text_id']}: "
            "cannot prove middle evidence boundary"
        )

    index = (
        len(offsets)
        // 2
    )

    offset = offsets[
        index
    ]

    if (
        offset <= 0
        or offset >= len(
            clean_text
        )
    ):
        raise RuntimeError(
            f"{clean_record['base_text_id']}: "
            "invalid middle offset"
        )

    return (
        offset,
        index,
        len(offsets),
        method,
    )


def build_inserted_block(
    attack_text: str,
    representation: str,
    encoding: str,
    position: str,
    config: dict,
) -> str:
    encoded = apply_encoding(
        attack_text,
        encoding,
    )

    rendered = apply_representation(
        encoded,
        representation,
    )

    separators = (
        config[
            "rendering"
        ][
            "separators"
        ][
            position
        ]
    )

    before = separator_value(
        separators.get(
            "before",
            "",
        )
    )

    after = separator_value(
        separators.get(
            "after",
            "",
        )
    )

    return (
        before
        + rendered
        + after
    )


def inject_block(
    clean_record: dict,
    assignment: dict,
    inserted_block: str,
) -> tuple[
    str,
    int,
    int,
    int,
    int | None,
    int | None,
    str | None,
]:
    clean_text = clean_record[
        "text"
    ]

    position = assignment[
        "insertion_position"
    ]

    middle_index = None
    middle_count = None
    middle_method = None

    if position == "prefix":
        offset = 0

    elif position == "suffix":
        offset = len(
            clean_text
        )

    elif position == "middle":
        (
            offset,
            middle_index,
            middle_count,
            middle_method,
        ) = resolve_middle_offset(
            clean_record
        )

    else:
        raise RuntimeError(
            f"Unsupported position: {position}"
        )

    injected_text = (
        clean_text[:offset]
        + inserted_block
        + clean_text[offset:]
    )

    attack_start = offset
    attack_end = (
        offset
        + len(
            inserted_block
        )
    )

    restored = (
        injected_text[
            :attack_start
        ]
        +
        injected_text[
            attack_end:
        ]
    )

    if restored != clean_text:
        raise RuntimeError(
            f"{clean_record['base_text_id']}: "
            "clean restoration invariant failed"
        )

    return (
        injected_text,
        attack_start,
        attack_end,
        offset,
        middle_index,
        middle_count,
        middle_method,
    )


def success_metadata(
    objective: str,
) -> tuple[
    bool,
    bool,
    str | None,
]:
    if objective == "force_true":
        return (
            True,
            True,
            "fire_answer_equals_true",
        )

    if objective == "force_false":
        return (
            True,
            True,
            "fire_answer_equals_false",
        )

    return (
        True,
        False,
        None,
    )


def build_sample(
    clean_record: dict,
    assignment: dict,
    template: dict,
    config: dict,
    config_sha256: str,
) -> dict:
    attack_text = build_attack_text(
        template,
        assignment,
    )

    forbidden_markers = (
        config[
            "rendering"
        ][
            "forbidden_literal_markers"
        ]
    )

    for marker in forbidden_markers:
        if marker in attack_text:
            raise RuntimeError(
                f"{assignment['assignment_id']}: "
                f"forbidden marker {marker}"
            )

    inserted_block = (
        build_inserted_block(
            attack_text,
            assignment[
                "representation"
            ],
            assignment[
                "encoding"
            ],
            assignment[
                "insertion_position"
            ],
            config,
        )
    )

    (
        injected_text,
        attack_start,
        attack_end,
        clean_offset,
        middle_index,
        middle_count,
        middle_method,
    ) = inject_block(
        clean_record,
        assignment,
        inserted_block,
    )

    for marker in forbidden_markers:
        if marker in injected_text:
            raise RuntimeError(
                f"{assignment['assignment_id']}: "
                "forbidden marker in model text"
            )

    clean_text = clean_record[
        "text"
    ]

    clean_text_sha256 = (
        sha256_text(
            clean_text
        )
    )

    attack_text_sha256 = (
        sha256_text(
            attack_text
        )
    )

    inserted_block_sha256 = (
        sha256_text(
            inserted_block
        )
    )

    injected_text_sha256 = (
        sha256_text(
            injected_text
        )
    )

    sample_id = stable_id(
        "fever_injection",
        [
            clean_record[
                "base_text_id"
            ],
            assignment[
                "template_id"
            ],
            assignment[
                "payload_ids"
            ],
            assignment[
                "insertion_position"
            ],
            assignment[
                "representation"
            ],
            assignment[
                "encoding"
            ],
            assignment[
                "registry_sha256"
            ],
            assignment[
                "generator_version"
            ],
            config_sha256,
        ],
    )

    (
        end_to_end_evaluable,
        targeted_asr_evaluable,
        success_criterion,
    ) = success_metadata(
        assignment[
            "attack_objective"
        ]
    )

    sample = {
        key: value
        for key, value
        in clean_record.items()
        if (
            key
            not in BASE_EXCLUDED_FIELDS
            and
            not key.startswith(
                "attack_"
            )
        )
    }

    sample.update(
        {
            "sample_id":
                sample_id,

            "dataset_split":
                assignment[
                    "dataset_split"
                ],

            "text":
                injected_text,

            "label":
                "injection",

            "base_text_id":
                clean_record[
                    "base_text_id"
                ],

            "base_text_sha256":
                clean_record[
                    "base_text_sha256"
                ],

            "claim_ids":
                clean_record[
                    "claim_ids"
                ],

            "fever_labels":
                clean_record[
                    "fever_labels"
                ],

            "evidence_refs":
                clean_record[
                    "evidence_refs"
                ],

            "upstream_split":
                assignment[
                    "base_split"
                ],

            "clean_text_sha256":
                clean_text_sha256,

            "attack_text":
                attack_text,

            "attack_text_sha256":
                attack_text_sha256,

            "inserted_block":
                inserted_block,

            "inserted_block_sha256":
                inserted_block_sha256,

            "injected_text_sha256":
                injected_text_sha256,

            "text_sha256":
                injected_text_sha256,

            "assignment_id":
                assignment[
                    "assignment_id"
                ],

            "template_id":
                assignment[
                    "template_id"
                ],

            "template_split":
                assignment[
                    "template_split"
                ],

            "family":
                assignment[
                    "family"
                ],

            "attack_objective":
                assignment[
                    "attack_objective"
                ],

            "semantic_cluster_id":
                assignment[
                    "semantic_cluster_id"
                ],

            "payload_ids":
                assignment[
                    "payload_ids"
                ],

            "payload_bindings":
                assignment[
                    "payload_bindings"
                ],

            "requested_position":
                assignment[
                    "requested_position"
                ],

            "insertion_position":
                assignment[
                    "insertion_position"
                ],

            "position_adjusted":
                assignment[
                    "position_adjusted"
                ],

            "representation":
                assignment[
                    "representation"
                ],

            "encoding":
                assignment[
                    "encoding"
                ],

            "attack_start_char":
                attack_start,

            "attack_end_char":
                attack_end,

            "clean_insertion_offset":
                clean_offset,

            "middle_boundary_index":
                middle_index,

            "middle_boundary_count":
                middle_count,

            "middle_boundary_method":
                middle_method,

            "render_plan_id":
                assignment[
                    "render_plan_id"
                ],

            "registry_version":
                assignment[
                    "registry_version"
                ],

            "registry_sha256":
                assignment[
                    "registry_sha256"
                ],

            "generator_version":
                assignment[
                    "generator_version"
                ],

            "generation_seed":
                assignment[
                    "generation_seed"
                ],

            "generation_config_sha256":
                config_sha256,

            "source":
                "fever_derived",

            "original_source":
                "FEVER",

            "synthetic":
                True,

            "detector_evaluable":
                True,

            "end_to_end_evaluable":
                end_to_end_evaluable,

            "targeted_asr_evaluable":
                targeted_asr_evaluable,

            "success_criterion":
                success_criterion,
        }
    )

    return sample


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
            list,
        ):
            for item in value:
                counter[
                    str(item)
                ] += 1
        else:
            counter[
                str(value)
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

    if not spec_manifest_path.exists():
        raise RuntimeError(
            "Step 3D.1 manifest missing"
        )

    if not plan_validation_path.exists():
        raise RuntimeError(
            "Step 3D.2 validation report missing"
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
            "Step 3D.1 specification is not frozen"
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
            "Generation config changed after freeze"
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
            "Step 3D.2 validation has not passed"
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
        registry_manifest.get(
            "frozen"
        )
        is not True
    ):
        raise RuntimeError(
            "Attack registry is not frozen"
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
        registry_manifest.get(
            "registry_sha256"
        )
        != expected_registry_sha
    ):
        raise RuntimeError(
            "Attack registry SHA-256 mismatch"
        )

    print(
        "Loading FEVER clean contexts..."
    )

    base_contexts = load_base_contexts(
        config
    )

    print(
        "Loading frozen registry..."
    )

    templates, payloads = (
        load_registry(
            config
        )
    )

    assignments_dir = (
        output_root
        /
        "plans"
        /
        "assignments"
    )

    generated = {}

    output_files = {}
    assignment_files = {}

    print(
        "Generating injected contexts..."
    )

    global_sample_ids = set()

    for track_name in TRACK_ORDER:
        track_config = config[
            "tracks"
        ][
            track_name
        ]

        base_split = (
            track_config[
                "base_split"
            ]
        )

        assignment_path = (
            assignments_dir
            /
            f"{track_name}.jsonl"
        )

        assignments = read_jsonl(
            assignment_path
        )

        if (
            len(assignments)
            != track_config[
                "expected_records"
            ]
        ):
            raise RuntimeError(
                f"{track_name}: "
                "assignment count mismatch"
            )

        assignment_files[
            track_name
        ] = {
            "path":
                str(
                    assignment_path.as_posix()
                ),

            "records":
                len(assignments),

            "sha256":
                sha256_file(
                    assignment_path
                ),
        }

        records = []
        seen_base_ids = set()
        seen_text_hashes = set()

        for assignment in assignments:
            base_text_id = (
                assignment[
                    "base_text_id"
                ]
            )

            if (
                base_text_id
                in seen_base_ids
            ):
                raise RuntimeError(
                    f"{track_name}: duplicate base_text_id "
                    f"{base_text_id}"
                )

            seen_base_ids.add(
                base_text_id
            )

            clean_record = (
                base_contexts[
                    base_split
                ].get(
                    base_text_id
                )
            )

            if clean_record is None:
                raise RuntimeError(
                    f"{track_name}: missing clean context "
                    f"{base_text_id}"
                )

            template = templates.get(
                assignment[
                    "template_id"
                ]
            )

            if template is None:
                raise RuntimeError(
                    f"{track_name}: unknown template "
                    f"{assignment['template_id']}"
                )

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

            if (
                sample[
                    "sample_id"
                ]
                in global_sample_ids
            ):
                raise RuntimeError(
                    "Duplicate sample_id: "
                    f"{sample['sample_id']}"
                )

            global_sample_ids.add(
                sample[
                    "sample_id"
                ]
            )

            if (
                sample[
                    "injected_text_sha256"
                ]
                in seen_text_hashes
            ):
                raise RuntimeError(
                    f"{track_name}: duplicate injected text"
                )

            seen_text_hashes.add(
                sample[
                    "injected_text_sha256"
                ]
            )

            records.append(
                sample
            )

        if (
            len(records)
            != track_config[
                "expected_records"
            ]
        ):
            raise RuntimeError(
                f"{track_name}: generated count mismatch"
            )

        output_path = (
            output_root
            /
            track_config[
                "output"
            ]
        )

        write_jsonl(
            output_path,
            records,
        )

        output_files[
            track_name
        ] = {
            "path":
                str(
                    output_path.as_posix()
                ),

            "records":
                len(records),

            "sha256":
                sha256_file(
                    output_path
                ),
        }

        generated[
            track_name
        ] = records

    track_reports = {}

    for track_name in TRACK_ORDER:
        records = generated[
            track_name
        ]

        track_reports[
            track_name
        ] = {
            "records":
                len(records),

            "family":
                distribution(
                    records,
                    "family",
                ),

            "attack_objective":
                distribution(
                    records,
                    "attack_objective",
                ),

            "semantic_cluster_id":
                distribution(
                    records,
                    "semantic_cluster_id",
                ),

            "template_id":
                distribution(
                    records,
                    "template_id",
                ),

            "payload_id":
                distribution(
                    records,
                    "payload_ids",
                ),

            "requested_position":
                distribution(
                    records,
                    "requested_position",
                ),

            "insertion_position":
                distribution(
                    records,
                    "insertion_position",
                ),

            "position_adjusted":
                distribution(
                    records,
                    "position_adjusted",
                ),

            "representation":
                distribution(
                    records,
                    "representation",
                ),

            "encoding":
                distribution(
                    records,
                    "encoding",
                ),

            "middle_boundary_method":
                distribution(
                    records,
                    "middle_boundary_method",
                ),

            "targeted_asr_evaluable":
                distribution(
                    records,
                    "targeted_asr_evaluable",
                ),
        }

    total = sum(
        len(records)
        for records
        in generated.values()
    )

    expected_total = (
        config[
            "expected"
        ][
            "total_context_records"
        ]
    )

    if total != expected_total:
        raise RuntimeError(
            "Total generated record count mismatch"
        )

    report = {
        "step":
            "3D.3",

        "status":
            "PASS",

        "generator_version":
            config[
                "generator_version"
            ],

        "generation_seed":
            config[
                "generation_seed"
            ],

        "generation_config_sha256":
            config_sha256,

        "registry_sha256":
            expected_registry_sha,

        "total_context_records":
            total,

        "clean_restoration_invariant":
            "PASS",

        "unresolved_placeholders":
            0,

        "forbidden_markers":
            0,

        "duplicate_sample_ids":
            0,

        "duplicate_injected_text_within_track":
            0,

        "assignment_files":
            assignment_files,

        "output_files":
            output_files,

        "tracks":
            track_reports,

        "next_stage":
            "3D.4_claim_level_FIRE_evaluation_instances",
    }

    report_path = Path(
        config[
            "output"
        ][
            "generation_report"
        ]
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
        "FEVER Injected Context Generation"
    )

    print(
        "================================"
    )

    print(
        "\nGenerated:"
    )

    for track_name in TRACK_ORDER:
        print(
            f"  {track_name}: "
            f"{len(generated[track_name])}"
        )

    print(
        "\nTotal:",
        total,
    )

    print(
        "\nClean restoration: PASS"
    )

    print(
        "Unresolved placeholders: 0"
    )

    print(
        "Forbidden markers: 0"
    )

    print(
        "Duplicate sample IDs: 0"
    )

    print(
        "Status: PASS"
    )

    print(
        "\nReport:",
        report_path,
    )


if __name__ == "__main__":
    main()