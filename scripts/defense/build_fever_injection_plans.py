from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import product
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


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
                record = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: {error}"
                ) from error

            records.append(record)

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


def seeded_order_key(
    base_split: str,
    base_text_id: str,
    seed: int,
) -> str:
    return sha256_text(
        f"{seed}\0{base_split}\0{base_text_id}"
    )


def load_base_contexts(
    config: dict,
) -> dict[str, list[dict]]:
    result = {}
    global_ids = {}

    for split, info in (
        config[
            "inputs"
        ][
            "fever_benign"
        ][
            "splits"
        ].items()
    ):
        path = Path(
            info[
                "path"
            ]
        )

        records = read_jsonl(
            path
        )

        if (
            len(records)
            != info[
                "expected_records"
            ]
        ):
            raise RuntimeError(
                f"{split}: expected "
                f"{info['expected_records']} records, "
                f"got {len(records)}"
            )

        seen = set()

        for record in records:
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
                raise RuntimeError(
                    f"{split}: invalid base_text_id"
                )

            if base_text_id in seen:
                raise RuntimeError(
                    f"{split}: duplicate base_text_id "
                    f"{base_text_id}"
                )

            seen.add(
                base_text_id
            )

            if (
                base_text_id
                in global_ids
            ):
                raise RuntimeError(
                    "Cross-split base_text_id leakage: "
                    f"{base_text_id} in "
                    f"{global_ids[base_text_id]} "
                    f"and {split}"
                )

            global_ids[
                base_text_id
            ] = split

            context_size = record.get(
                "context_size"
            )

            if (
                not isinstance(
                    context_size,
                    int,
                )
                or context_size < 1
            ):
                raise RuntimeError(
                    f"{split}:{base_text_id}: "
                    "invalid context_size"
                )

            text = record.get(
                "text"
            )

            if (
                not isinstance(
                    text,
                    str,
                )
                or not text
            ):
                raise RuntimeError(
                    f"{split}:{base_text_id}: "
                    "invalid text"
                )

        result[
            split
        ] = records

    return result


def load_registry(
    config: dict,
) -> tuple[
    dict[str, list[dict]],
    list[dict],
]:
    registry = config[
        "inputs"
    ][
        "attack_registry"
    ]

    templates = {}

    for split, path_string in (
        registry[
            "template_splits"
        ].items()
    ):
        templates[
            split
        ] = read_jsonl(
            Path(
                path_string
            )
        )

    payloads = read_jsonl(
        Path(
            registry[
                "payload_registry"
            ]
        )
    )

    return templates, payloads


def build_payload_index(
    payloads: list[dict],
) -> dict:
    result = defaultdict(
        list
    )

    for payload in payloads:
        placeholder = payload[
            "placeholder"
        ]

        for objective in payload[
            "compatible_objectives"
        ]:
            result[
                (
                    placeholder,
                    objective,
                )
            ].append(
                payload
            )

    for key in result:
        result[
            key
        ].sort(
            key=lambda item:
                item[
                    "payload_id"
                ]
        )

    return result


class HierarchicalAssigner:
    def __init__(
        self,
        templates: list[dict],
        payload_index: dict,
    ):
        self.payload_index = (
            payload_index
        )

        self.tree = defaultdict(
            lambda:
                defaultdict(
                    lambda:
                        defaultdict(
                            list
                        )
                )
        )

        for template in templates:
            if (
                template.get(
                    "enabled"
                )
                is not True
            ):
                continue

            family = template[
                "family"
            ]

            objective = template[
                "attack_objective"
            ]

            cluster = template[
                "semantic_cluster_id"
            ]

            self.tree[
                family
            ][
                objective
            ][
                cluster
            ].append(
                template
            )

        for family in self.tree:
            for objective in (
                self.tree[
                    family
                ]
            ):
                for cluster in (
                    self.tree[
                        family
                    ][
                        objective
                    ]
                ):
                    self.tree[
                        family
                    ][
                        objective
                    ][
                        cluster
                    ].sort(
                        key=lambda item:
                            item[
                                "template_id"
                            ]
                    )

        self.families = sorted(
            self.tree
        )

        if not self.families:
            raise RuntimeError(
                "No enabled templates"
            )

        self.family_counter = 0

        self.objective_counter = (
            defaultdict(
                int
            )
        )

        self.cluster_counter = (
            defaultdict(
                int
            )
        )

        self.template_counter = (
            defaultdict(
                int
            )
        )

        self.payload_counter = (
            defaultdict(
                int
            )
        )

    def get_templates_for_family_objective(
        self,
        family: str,
        objective: str,
    ) -> list[dict]:
        result = []

        clusters = sorted(
            self.tree[
                family
            ][
                objective
            ]
        )

        for cluster in clusters:
            result.extend(
                self.tree[
                    family
                ][
                    objective
                ][
                    cluster
                ]
            )

        return result

    def assign(
        self,
    ) -> dict:
        family = self.families[
            self.family_counter
            % len(
                self.families
            )
        ]

        self.family_counter += 1

        objectives = sorted(
            self.tree[
                family
            ]
        )

        objective = objectives[
            self.objective_counter[
                family
            ]
            % len(
                objectives
            )
        ]

        self.objective_counter[
            family
        ] += 1

        clusters = sorted(
            self.tree[
                family
            ][
                objective
            ]
        )

        cluster_key = (
            family,
            objective,
        )

        cluster = clusters[
            self.cluster_counter[
                cluster_key
            ]
            % len(
                clusters
            )
        ]

        self.cluster_counter[
            cluster_key
        ] += 1

        templates = (
            self.tree[
                family
            ][
                objective
            ][
                cluster
            ]
        )

        template_key = (
            family,
            objective,
            cluster,
        )

        template = templates[
            self.template_counter[
                template_key
            ]
            % len(
                templates
            )
        ]

        self.template_counter[
            template_key
        ] += 1

        bindings = {}
        payload_ids = []

        for placeholder in (
            template.get(
                "placeholders",
                [],
            )
        ):
            candidates = (
                self.payload_index.get(
                    (
                        placeholder,
                        objective,
                    ),
                    [],
                )
            )

            if not candidates:
                raise RuntimeError(
                    "No compatible payload for "
                    f"{template['template_id']} "
                    f"placeholder={placeholder} "
                    f"objective={objective}"
                )

            payload_key = (
                template[
                    "template_id"
                ],
                placeholder,
            )

            payload = candidates[
                self.payload_counter[
                    payload_key
                ]
                % len(
                    candidates
                )
            ]

            self.payload_counter[
                payload_key
            ] += 1

            bindings[
                placeholder
            ] = {
                "payload_id":
                    payload[
                        "payload_id"
                    ],

                "value":
                    payload[
                        "value"
                    ],
            }

            payload_ids.append(
                payload[
                    "payload_id"
                ]
            )

        return {
            "template":
                template,

            "payload_bindings":
                bindings,

            "payload_ids":
                payload_ids,
        }


def build_render_plans(
    base_contexts: dict[
        str,
        list[dict],
    ],
    config: dict,
    config_sha256: str,
) -> tuple[
    list[dict],
    dict[
        tuple[str, str],
        dict,
    ],
]:
    render_config = config[
        "render_plan"
    ]

    position_config = (
        render_config[
            "position"
        ]
    )

    requested_cycle = (
        position_config[
            "requested_cycle"
        ]
    )

    fallback_cycle = (
        position_config[
            "ineligible_requested_position"
        ][
            "fallback_cycle"
        ]
    )

    representations = (
        render_config[
            "representation"
        ][
            "allowed"
        ]
    )

    encodings = (
        render_config[
            "encoding"
        ][
            "allowed"
        ]
    )

    seed = config[
        "generation_seed"
    ]

    records = []
    index = {}

    for base_split in (
        "train",
        "paper_dev",
        "paper_test",
    ):
        contexts = list(
            base_contexts[
                base_split
            ]
        )

        contexts.sort(
            key=lambda record: (
                seeded_order_key(
                    base_split,
                    record[
                        "base_text_id"
                    ],
                    seed,
                ),
                record[
                    "base_text_id"
                ],
            )
        )

        fallback_counter = 0

        for rank, context in enumerate(
            contexts
        ):
            base_text_id = context[
                "base_text_id"
            ]

            context_size = context[
                "context_size"
            ]

            requested_position = (
                requested_cycle[
                    rank
                    % len(
                        requested_cycle
                    )
                ]
            )

            if context_size == 1:
                eligible_positions = set(
                    position_config[
                        "singleton_context"
                    ][
                        "eligible"
                    ]
                )

            else:
                eligible_positions = set(
                    position_config[
                        "multi_sentence_context"
                    ][
                        "eligible"
                    ]
                )

            if (
                requested_position
                in eligible_positions
            ):
                insertion_position = (
                    requested_position
                )

                position_adjusted = (
                    False
                )

            else:
                valid_fallbacks = [
                    position
                    for position
                    in fallback_cycle
                    if (
                        position
                        in eligible_positions
                    )
                ]

                if not valid_fallbacks:
                    raise RuntimeError(
                        "No valid fallback position "
                        f"for {base_text_id}"
                    )

                insertion_position = (
                    valid_fallbacks[
                        fallback_counter
                        % len(
                            valid_fallbacks
                        )
                    ]
                )

                fallback_counter += 1

                position_adjusted = (
                    True
                )

            representation = (
                representations[
                    rank
                    % len(
                        representations
                    )
                ]
            )

            encoding = (
                encodings[
                    rank
                    % len(
                        encodings
                    )
                ]
            )

            render_plan_id = (
                stable_id(
                    "fever_render_plan",
                    [
                        base_text_id,
                        base_split,
                        insertion_position,
                        representation,
                        encoding,
                        seed,
                        config_sha256,
                    ],
                )
            )

            record = {
                "render_plan_id":
                    render_plan_id,

                "base_text_id":
                    base_text_id,

                "base_split":
                    base_split,

                "context_size":
                    context_size,

                "assignment_rank":
                    rank,

                "requested_position":
                    requested_position,

                "insertion_position":
                    insertion_position,

                "position_adjusted":
                    position_adjusted,

                "representation":
                    representation,

                "encoding":
                    encoding,

                "generation_seed":
                    seed,

                "generation_config_sha256":
                    config_sha256,
            }

            records.append(
                record
            )

            index[
                (
                    base_split,
                    base_text_id,
                )
            ] = record

    records.sort(
        key=lambda record: (
            record[
                "base_split"
            ],
            record[
                "base_text_id"
            ],
        )
    )

    return records, index


def validate_template_render_compatibility(
    template: dict,
    render_plan: dict,
) -> None:
    if (
        render_plan[
            "insertion_position"
        ]
        not in template[
            "allowed_positions"
        ]
    ):
        raise RuntimeError(
            f"{template['template_id']}: "
            "position not allowed"
        )

    if (
        render_plan[
            "representation"
        ]
        not in template[
            "allowed_representations"
        ]
    ):
        raise RuntimeError(
            f"{template['template_id']}: "
            "representation not allowed"
        )

    if (
        render_plan[
            "encoding"
        ]
        not in template[
            "allowed_encodings"
        ]
    ):
        raise RuntimeError(
            f"{template['template_id']}: "
            "encoding not allowed"
        )


def render_attack_for_collision(
    template: dict,
    payload_bindings: dict,
    representation: str,
    encoding: str,
) -> str:
    attack_text = template[
        "template"
    ]

    for placeholder in (
        template.get(
            "placeholders",
            [],
        )
    ):
        attack_text = (
            attack_text.replace(
                "{{"
                + placeholder
                + "}}",
                payload_bindings[
                    placeholder
                ][
                    "value"
                ],
            )
        )

    if encoding != "plain":
        raise RuntimeError(
            "Unsupported encoding "
            f"in Step 3D.2: {encoding}"
        )

    if (
        representation
        == "plain_text"
    ):
        return attack_text

    if representation == "markdown":
        return "\n".join(
            (
                f"> {line}"
                if line
                else ">"
            )
            for line in (
                attack_text.split(
                    "\n"
                )
            )
        )

    raise RuntimeError(
        "Unsupported representation "
        f"in Step 3D.2: {representation}"
    )


def build_assignment_collision_key(
    clean_record: dict,
    template: dict,
    payload_bindings: dict,
    render_plan: dict,
) -> str:
    rendered_attack = (
        render_attack_for_collision(
            template,
            payload_bindings,
            render_plan[
                "representation"
            ],
            render_plan[
                "encoding"
            ],
        )
    )

    payload = [
        clean_record[
            "text"
        ],
        rendered_attack,
        render_plan[
            "insertion_position"
        ],
        render_plan[
            "representation"
        ],
        render_plan[
            "encoding"
        ],
    ]

    return sha256_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def payload_variants(
    template: dict,
    objective: str,
    payload_index: dict,
):
    placeholders = (
        template.get(
            "placeholders",
            [],
        )
    )

    if not placeholders:
        yield {
            "payload_bindings": {},
            "payload_ids": [],
        }

        return

    candidate_lists = []

    for placeholder in placeholders:
        candidates = (
            payload_index.get(
                (
                    placeholder,
                    objective,
                ),
                [],
            )
        )

        if not candidates:
            raise RuntimeError(
                "No compatible payload for "
                f"{template['template_id']} "
                f"placeholder={placeholder} "
                f"objective={objective}"
            )

        candidate_lists.append(
            candidates
        )

    for combination in product(
        *candidate_lists
    ):
        bindings = {}
        payload_ids = []

        for placeholder, payload in zip(
            placeholders,
            combination,
        ):
            bindings[
                placeholder
            ] = {
                "payload_id":
                    payload[
                        "payload_id"
                    ],

                "value":
                    payload[
                        "value"
                    ],
            }

            payload_ids.append(
                payload[
                    "payload_id"
                ]
            )

        yield {
            "payload_bindings":
                bindings,

            "payload_ids":
                payload_ids,
        }


def ordered_collision_templates(
    assigner: HierarchicalAssigner,
    current_template: dict,
) -> list[dict]:
    family = current_template[
        "family"
    ]

    objective = current_template[
        "attack_objective"
    ]

    current_cluster = (
        current_template[
            "semantic_cluster_id"
        ]
    )

    all_templates = (
        assigner.get_templates_for_family_objective(
            family,
            objective,
        )
    )

    return sorted(
        all_templates,
        key=lambda template: (
            (
                template[
                    "semantic_cluster_id"
                ]
                != current_cluster
            ),
            (
                template[
                    "template_id"
                ]
                != current_template[
                    "template_id"
                ]
            ),
            template[
                "semantic_cluster_id"
            ],
            template[
                "template_id"
            ],
        ),
    )


def repair_assignment_collision(
    clean_record: dict,
    chosen: dict,
    render_plan: dict,
    payload_index: dict,
    assigner: HierarchicalAssigner,
    seen_collision_keys: set[str],
) -> tuple[
    dict,
    bool,
    str | None,
]:
    current_template = chosen[
        "template"
    ]

    current_key = (
        build_assignment_collision_key(
            clean_record,
            current_template,
            chosen[
                "payload_bindings"
            ],
            render_plan,
        )
    )

    if (
        current_key
        not in seen_collision_keys
    ):
        return (
            chosen,
            False,
            None,
        )

    current_family = (
        current_template[
            "family"
        ]
    )

    current_objective = (
        current_template[
            "attack_objective"
        ]
    )

    current_cluster = (
        current_template[
            "semantic_cluster_id"
        ]
    )

    current_template_id = (
        current_template[
            "template_id"
        ]
    )

    current_payload_ids = tuple(
        chosen[
            "payload_ids"
        ]
    )

    candidates = (
        ordered_collision_templates(
            assigner,
            current_template,
        )
    )

    for candidate_template in (
        candidates
    ):
        if (
            candidate_template[
                "family"
            ]
            != current_family
        ):
            continue

        if (
            candidate_template[
                "attack_objective"
            ]
            != current_objective
        ):
            continue

        try:
            validate_template_render_compatibility(
                candidate_template,
                render_plan,
            )

        except RuntimeError:
            continue

        for variant in payload_variants(
            candidate_template,
            current_objective,
            payload_index,
        ):
            candidate_template_id = (
                candidate_template[
                    "template_id"
                ]
            )

            candidate_payload_ids = tuple(
                variant[
                    "payload_ids"
                ]
            )

            if (
                candidate_template_id
                == current_template_id
                and
                candidate_payload_ids
                == current_payload_ids
            ):
                continue

            candidate_key = (
                build_assignment_collision_key(
                    clean_record,
                    candidate_template,
                    variant[
                        "payload_bindings"
                    ],
                    render_plan,
                )
            )

            if (
                candidate_key
                in seen_collision_keys
            ):
                continue

            candidate_cluster = (
                candidate_template[
                    "semantic_cluster_id"
                ]
            )

            if (
                candidate_template_id
                == current_template_id
            ):
                repair_mode = (
                    "payload"
                )

            elif (
                candidate_cluster
                == current_cluster
            ):
                repair_mode = (
                    "template_same_cluster"
                )

            else:
                repair_mode = (
                    "template_other_cluster"
                )

            repaired = {
                "template":
                    candidate_template,

                "payload_bindings":
                    variant[
                        "payload_bindings"
                    ],

                "payload_ids":
                    variant[
                        "payload_ids"
                    ],
            }

            return (
                repaired,
                True,
                repair_mode,
            )

    raise RuntimeError(
        "Unable to resolve assignment collision: "
        f"base_text_id="
        f"{clean_record['base_text_id']} "
        f"family={current_family} "
        f"objective={current_objective} "
        f"cluster={current_cluster}"
    )


def build_track_assignments(
    track_name: str,
    track_config: dict,
    base_contexts: dict,
    templates_by_split: dict,
    payload_index: dict,
    render_index: dict,
    config: dict,
    config_sha256: str,
) -> list[dict]:
    base_split = track_config[
        "base_split"
    ]

    template_split = (
        track_config[
            "template_split"
        ]
    )

    seed = config[
        "generation_seed"
    ]

    registry = config[
        "inputs"
    ][
        "attack_registry"
    ]

    generator_version = (
        config[
            "generator_version"
        ]
    )

    contexts = list(
        base_contexts[
            base_split
        ]
    )

    contexts.sort(
        key=lambda record: (
            seeded_order_key(
                base_split,
                record[
                    "base_text_id"
                ],
                seed,
            ),
            record[
                "base_text_id"
            ],
        )
    )

    assigner = (
        HierarchicalAssigner(
            templates_by_split[
                template_split
            ],
            payload_index,
        )
    )

    assignments = []

    seen_collision_keys = set()

    for rank, context in enumerate(
        contexts
    ):
        base_text_id = context[
            "base_text_id"
        ]

        render_plan = (
            render_index[
                (
                    base_split,
                    base_text_id,
                )
            ]
        )

        chosen = (
            assigner.assign()
        )

        (
            chosen,
            collision_repaired,
            collision_repair_mode,
        ) = repair_assignment_collision(
            context,
            chosen,
            render_plan,
            payload_index,
            assigner,
            seen_collision_keys,
        )

        template = chosen[
            "template"
        ]

        validate_template_render_compatibility(
            template,
            render_plan,
        )

        collision_key = (
            build_assignment_collision_key(
                context,
                template,
                chosen[
                    "payload_bindings"
                ],
                render_plan,
            )
        )

        if (
            collision_key
            in seen_collision_keys
        ):
            raise RuntimeError(
                "Unresolved assignment collision: "
                f"{track_name}:"
                f"{base_text_id}"
            )

        seen_collision_keys.add(
            collision_key
        )

        payload_ids = chosen[
            "payload_ids"
        ]

        assignment_id = (
            stable_id(
                "fever_assignment",
                [
                    track_name,
                    base_text_id,
                    template[
                        "template_id"
                    ],
                    payload_ids,
                    render_plan[
                        "render_plan_id"
                    ],
                    registry[
                        "registry_sha256"
                    ],
                    generator_version,
                    config_sha256,
                ],
            )
        )

        assignments.append(
            {
                "assignment_id":
                    assignment_id,

                "dataset_split":
                    track_name,

                "base_split":
                    base_split,

                "base_text_id":
                    base_text_id,

                "context_size":
                    context[
                        "context_size"
                    ],

                "assignment_rank":
                    rank,

                "render_plan_id":
                    render_plan[
                        "render_plan_id"
                    ],

                "requested_position":
                    render_plan[
                        "requested_position"
                    ],

                "insertion_position":
                    render_plan[
                        "insertion_position"
                    ],

                "position_adjusted":
                    render_plan[
                        "position_adjusted"
                    ],

                "representation":
                    render_plan[
                        "representation"
                    ],

                "encoding":
                    render_plan[
                        "encoding"
                    ],

                "template_id":
                    template[
                        "template_id"
                    ],

                "template_split":
                    template_split,

                "family":
                    template[
                        "family"
                    ],

                "attack_objective":
                    template[
                        "attack_objective"
                    ],

                "semantic_cluster_id":
                    template[
                        "semantic_cluster_id"
                    ],

                "payload_ids":
                    payload_ids,

                "payload_bindings":
                    chosen[
                        "payload_bindings"
                    ],

                "collision_repaired":
                    collision_repaired,

                "collision_repair_mode":
                    collision_repair_mode,

                "registry_version":
                    registry[
                        "version"
                    ],

                "registry_sha256":
                    registry[
                        "registry_sha256"
                    ],

                "generator_version":
                    generator_version,

                "generation_seed":
                    seed,

                "generation_config_sha256":
                    config_sha256,
            }
        )

    assignments.sort(
        key=lambda record:
            record[
                "base_text_id"
            ]
    )

    expected = track_config[
        "expected_records"
    ]

    if (
        len(assignments)
        != expected
    ):
        raise RuntimeError(
            f"{track_name}: expected "
            f"{expected} assignments, "
            f"got {len(assignments)}"
        )

    return assignments


def distribution(
    records: list[dict],
    field: str,
) -> dict:
    values = Counter()

    for record in records:
        value = record.get(
            field
        )

        if isinstance(
            value,
            list,
        ):
            for item in value:
                values[
                    str(item)
                ] += 1

        else:
            values[
                str(value)
            ] += 1

    return dict(
        sorted(
            values.items()
        )
    )


def build_report(
    assignments: dict[
        str,
        list[dict],
    ],
    render_plans: list[dict],
    config_sha256: str,
) -> dict:
    tracks = {}

    total_collision_repairs = 0

    collision_repair_modes = (
        Counter()
    )

    for name, records in (
        assignments.items()
    ):
        collision_repairs = sum(
            1
            for record in records
            if (
                record.get(
                    "collision_repaired"
                )
                is True
            )
        )

        total_collision_repairs += (
            collision_repairs
        )

        for record in records:
            mode = record.get(
                "collision_repair_mode"
            )

            if mode is not None:
                collision_repair_modes[
                    mode
                ] += 1

        tracks[
            name
        ] = {
            "records":
                len(records),

            "collision_repairs":
                collision_repairs,

            "collision_repair_mode":
                distribution(
                    records,
                    "collision_repair_mode",
                ),

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
        }

    return {
        "step":
            "3D.2",

        "status":
            "BUILT",

        "generation_config_sha256":
            config_sha256,

        "render_plans":
            len(
                render_plans
            ),

        "assignment_records":
            sum(
                len(records)
                for records
                in assignments.values()
            ),

        "collision_repairs":
            total_collision_repairs,

        "collision_repair_modes":
            dict(
                sorted(
                    collision_repair_modes.items()
                )
            ),

        "tracks":
            tracks,
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

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    config = read_yaml(
        config_path
    )

    config_sha256 = (
        sha256_file(
            config_path
        )
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

    if (
        not spec_manifest_path.exists()
    ):
        raise RuntimeError(
            "3D.1 spec_manifest.json missing"
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
            "3D.1 specification is not frozen"
        )

    frozen_config_sha = (
        spec_manifest[
            "generation_config"
        ][
            "sha256"
        ]
    )

    if (
        config_sha256
        != frozen_config_sha
    ):
        raise RuntimeError(
            "Generation configuration changed "
            "after Step 3D.1 freeze"
        )

    registry_manifest_path = Path(
        config[
            "inputs"
        ][
            "attack_registry"
        ][
            "manifest"
        ]
    )

    registry_manifest = read_json(
        registry_manifest_path
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
        "Loading FEVER contexts..."
    )

    base_contexts = (
        load_base_contexts(
            config
        )
    )

    print(
        "Loading frozen attack registry..."
    )

    (
        templates_by_split,
        payloads,
    ) = load_registry(
        config
    )

    payload_index = (
        build_payload_index(
            payloads
        )
    )

    print(
        "Building shared render plans..."
    )

    (
        render_plans,
        render_index,
    ) = build_render_plans(
        base_contexts,
        config,
        config_sha256,
    )

    render_path = Path(
        config[
            "output"
        ][
            "render_plans"
        ]
    )

    write_jsonl(
        render_path,
        render_plans,
    )

    assignments_dir = (
        output_root
        /
        "plans"
        /
        "assignments"
    )

    assignments = {}

    print(
        "Building hierarchical assignments..."
    )

    for track_name in (
        TRACK_ORDER
    ):
        track_config = (
            config[
                "tracks"
            ][
                track_name
            ]
        )

        records = (
            build_track_assignments(
                track_name,
                track_config,
                base_contexts,
                templates_by_split,
                payload_index,
                render_index,
                config,
                config_sha256,
            )
        )

        assignments[
            track_name
        ] = records

        write_jsonl(
            assignments_dir
            /
            f"{track_name}.jsonl",
            records,
        )

    report = build_report(
        assignments,
        render_plans,
        config_sha256,
    )

    report_path = (
        output_root
        /
        "plan_report.json"
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
        "FEVER Injection Plans"
    )

    print(
        "================================"
    )

    print(
        "\nRender plans:",
        len(
            render_plans
        ),
    )

    print(
        "\nAssignments:"
    )

    for track_name in (
        TRACK_ORDER
    ):
        repairs = sum(
            1
            for record
            in assignments[
                track_name
            ]
            if (
                record.get(
                    "collision_repaired"
                )
                is True
            )
        )

        print(
            f"  {track_name}: "
            f"{len(assignments[track_name])} "
            f"(collision repairs: {repairs})"
        )

    total = sum(
        len(records)
        for records
        in assignments.values()
    )

    print(
        "\nTotal assignments:",
        total,
    )

    print(
        "Collision repairs:",
        report[
            "collision_repairs"
        ],
    )

    if (
        report[
            "collision_repair_modes"
        ]
    ):
        print(
            "Collision repair modes:"
        )

        for mode, count in (
            report[
                "collision_repair_modes"
            ].items()
        ):
            print(
                f"  {mode}: {count}"
            )

    print(
        "\nRender plans:",
        render_path,
    )

    print(
        "Assignments:",
        assignments_dir,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()