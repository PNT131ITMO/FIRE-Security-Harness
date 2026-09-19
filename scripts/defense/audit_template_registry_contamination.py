from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


TEMPLATE_FILES = (
    "train.jsonl",
    "validation.jsonl",
    "ood_template.jsonl",
    "ood_family.jsonl",
)

PLACEHOLDER_PATTERN = re.compile(
    r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}"
)

REFERENCE_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".txt",
    ".csv",
}

MIN_REFERENCE_LENGTH = 8


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
                    json.loads(line)
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"{path}:{line_number}: {error}"
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


def template_static_text(
    template: str,
) -> str:

    text = PLACEHOLDER_PATTERN.sub(
        " ",
        template,
    )

    return " ".join(
        text.split()
    )


def load_registry_items(
    registry_dir: Path,
) -> list[dict]:

    result = []

    for file_name in TEMPLATE_FILES:

        path = (
            registry_dir
            /
            file_name
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        split = path.stem

        for record in read_jsonl(
            path
        ):

            template = record.get(
                "template"
            )

            if not isinstance(
                template,
                str,
            ):

                continue

            result.append(
                {
                    "registry_item_id":
                        record.get(
                            "template_id"
                        ),

                    "registry_item_type":
                        "template",

                    "template_split":
                        split,

                    "family":
                        record.get(
                            "family"
                        ),

                    "semantic_cluster_id":
                        record.get(
                            "semantic_cluster_id"
                        ),

                    "attack_objective":
                        record.get(
                            "attack_objective"
                        ),

                    "variant":
                        "raw",

                    "text":
                        template,

                    "normalized_text":
                        normalize_text(
                            template
                        ),
                }
            )

            static_text = (
                template_static_text(
                    template
                )
            )

            if (
                static_text
                and
                normalize_text(
                    static_text
                )
                != normalize_text(
                    template
                )
            ):

                result.append(
                    {
                        "registry_item_id":
                            record.get(
                                "template_id"
                            ),

                        "registry_item_type":
                            "template",

                        "template_split":
                            split,

                        "family":
                            record.get(
                                "family"
                            ),

                        "semantic_cluster_id":
                            record.get(
                                "semantic_cluster_id"
                            ),

                        "attack_objective":
                            record.get(
                                "attack_objective"
                            ),

                        "variant":
                            "placeholder_stripped",

                        "text":
                            static_text,

                        "normalized_text":
                            normalize_text(
                                static_text
                            ),
                    }
                )

    payload_path = (
        registry_dir
        /
        "payload_registry.jsonl"
    )

    if not payload_path.exists():

        raise FileNotFoundError(
            payload_path
        )

    for record in read_jsonl(
        payload_path
    ):

        value = record.get(
            "value"
        )

        if not isinstance(
            value,
            str,
        ):

            continue

        result.append(
            {
                "registry_item_id":
                    record.get(
                        "payload_id"
                    ),

                "registry_item_type":
                    "payload",

                "template_split":
                    None,

                "family":
                    None,

                "semantic_cluster_id":
                    None,

                "attack_objective":
                    (
                        record.get(
                            "compatible_objectives"
                        )
                    ),

                "variant":
                    "raw",

                "text":
                    value,

                "normalized_text":
                    normalize_text(
                        value
                    ),
            }
        )

    return result


def classify_reference_group(
    path: Path,
) -> str:

    value = str(
        path
    ).casefold()

    if (
        "injecguard_clean"
        in value
    ):

        return "injecguard_clean"

    if (
        "injecguard_quarantine"
        in value
    ):

        return "injecguard_quarantine"

    if (
        "notinject"
        in value
    ):

        return "notinject"

    if (
        "bipia"
        in value
    ):

        if (
            "test"
            in value
        ):

            return "bipia_test"

        if (
            "train"
            in value
        ):

            return "bipia_train"

        return "bipia"

    return "benchmark_other"


def collect_string_values(
    value,
    path: str = "$",
):

    if isinstance(
        value,
        str,
    ):

        if len(
            value.strip()
        ) >= MIN_REFERENCE_LENGTH:

            yield (
                path,
                value,
            )

        return

    if isinstance(
        value,
        list,
    ):

        for index, item in enumerate(
            value
        ):

            yield from collect_string_values(
                item,
                f"{path}[{index}]",
            )

        return

    if isinstance(
        value,
        dict,
    ):

        for key, item in value.items():

            yield from collect_string_values(
                item,
                f"{path}.{key}",
            )


def load_json_reference(
    path: Path,
) -> list[dict]:

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    result = []

    for json_path, text in (
        collect_string_values(
            value
        )
    ):

        result.append(
            {
                "json_path":
                    json_path,

                "text":
                    text,
            }
        )

    return result


def load_jsonl_reference(
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

                value = json.loads(
                    line
                )

            except json.JSONDecodeError:

                continue

            for json_path, text in (
                collect_string_values(
                    value
                )
            ):

                result.append(
                    {
                        "json_path":
                            (
                                f"$line[{line_number}]"
                                f"{json_path[1:]}"
                            ),

                        "text":
                            text,
                    }
                )

    return result


def load_txt_reference(
    path: Path,
) -> list[dict]:

    result = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            text = line.strip()

            if len(
                text
            ) < MIN_REFERENCE_LENGTH:

                continue

            result.append(
                {
                    "json_path":
                        f"$line[{line_number}]",

                    "text":
                        text,
                }
            )

    return result


def load_csv_reference(
    path: Path,
) -> list[dict]:

    result = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):

            for key, value in row.items():

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and
                    len(
                        value.strip()
                    )
                    >= MIN_REFERENCE_LENGTH
                ):

                    result.append(
                        {
                            "json_path":
                                (
                                    f"$row[{row_number}]"
                                    f".{key}"
                                ),

                            "text":
                                value,
                        }
                    )

    return result


def load_reference_file(
    path: Path,
) -> list[dict]:

    suffix = (
        path.suffix.casefold()
    )

    if suffix == ".json":

        return load_json_reference(
            path
        )

    if suffix == ".jsonl":

        return load_jsonl_reference(
            path
        )

    if suffix == ".txt":

        return load_txt_reference(
            path
        )

    if suffix == ".csv":

        return load_csv_reference(
            path
        )

    return []


def load_reference_items(
    paths: list[Path],
) -> list[dict]:

    result = []
    seen = set()

    for root in paths:

        if not root.exists():

            continue

        if root.is_file():

            files = [
                root
            ]

        else:

            files = sorted(
                path
                for path
                in root.rglob("*")
                if (
                    path.is_file()
                    and
                    path.suffix.casefold()
                    in REFERENCE_EXTENSIONS
                )
            )

        for path in files:

            group = (
                classify_reference_group(
                    path
                )
            )

            try:

                records = (
                    load_reference_file(
                        path
                    )
                )

            except Exception as error:

                print(
                    "Skipping reference file:",
                    path,
                    error,
                )

                continue

            for record in records:

                normalized = (
                    normalize_text(
                        record["text"]
                    )
                )

                if not normalized:

                    continue

                key = (
                    group,
                    normalized,
                )

                if key in seen:

                    continue

                seen.add(
                    key
                )

                result.append(
                    {
                        "reference_group":
                            group,

                        "reference_file":
                            str(
                                path
                            ),

                        "reference_location":
                            record[
                                "json_path"
                            ],

                        "text":
                            record[
                                "text"
                            ],

                        "normalized_text":
                            normalized,

                        "normalized_sha256":
                            sha256_text(
                                normalized
                            ),
                    }
                )

    return result


def char_ngrams(
    text: str,
    n: int,
) -> set[str]:

    text = normalize_text(
        text
    )

    if len(
        text
    ) <= n:

        return {
            text
        }

    return {
        text[
            index:
            index + n
        ]
        for index
        in range(
            len(text)
            - n
            + 1
        )
    }


def jaccard_similarity(
    left: set[str],
    right: set[str],
) -> float:

    if (
        not left
        and
        not right
    ):

        return 1.0

    union = (
        left
        |
        right
    )

    if not union:

        return 0.0

    return (
        len(
            left
            &
            right
        )
        /
        len(
            union
        )
    )


def sequence_similarity(
    left: str,
    right: str,
) -> float:

    from difflib import SequenceMatcher

    return SequenceMatcher(
        None,
        left,
        right,
        autojunk=False,
    ).ratio()


def substring_match(
    left: str,
    right: str,
    minimum_chars: int,
    minimum_ratio: float,
) -> bool:

    if (
        len(left)
        < minimum_chars
        or
        len(right)
        < minimum_chars
    ):

        return False

    shorter = min(
        len(left),
        len(right),
    )

    longer = max(
        len(left),
        len(right),
    )

    ratio = (
        shorter
        /
        longer
    )

    if ratio < minimum_ratio:

        return False

    return (
        left in right
        or
        right in left
    )


def classify_severity(
    reference_group: str,
    match_type: str,
) -> tuple[str, bool]:

    bipia = (
        reference_group
        in {
            "bipia",
            "bipia_train",
            "bipia_test",
        }
    )

    if bipia:

        if match_type in {
            "exact_normalized",
            "substring",
        }:

            return (
                "critical",
                True,
            )

        if match_type == "near_duplicate":

            return (
                "high",
                True,
            )

        if match_type == "semantic_similarity":

            return (
                "high",
                False,
            )

    if (
        reference_group
        == "notinject"
    ):

        if match_type in {
            "exact_normalized",
            "substring",
            "near_duplicate",
        }:

            return (
                "high",
                False,
            )

        return (
            "medium",
            False,
        )

    if reference_group in {
        "injecguard_clean",
        "injecguard_quarantine",
    }:

        if match_type == "exact_normalized":

            return (
                "medium",
                False,
            )

        return (
            "informational",
            False,
        )

    if match_type == "exact_normalized":

        return (
            "medium",
            False,
        )

    return (
        "informational",
        False,
    )


def lexical_audit(
    registry_items: list[dict],
    reference_items: list[dict],
    ngram_size: int,
    jaccard_threshold: float,
    sequence_threshold: float,
    substring_min_chars: int,
    substring_min_ratio: float,
) -> list[dict]:

    matches = []

    reference_exact = {}

    for reference in (
        reference_items
    ):

        reference_exact.setdefault(
            reference[
                "normalized_text"
            ],
            [],
        ).append(
            reference
        )

    reference_ngrams = [
        char_ngrams(
            reference[
                "normalized_text"
            ],
            ngram_size,
        )
        for reference
        in reference_items
    ]

    for item in registry_items:

        normalized = item[
            "normalized_text"
        ]

        if not normalized:

            continue

        exact_matches = (
            reference_exact.get(
                normalized,
                [],
            )
        )

        exact_reference_keys = set()

        for reference in (
            exact_matches
        ):

            key = (
                reference[
                    "reference_file"
                ],
                reference[
                    "reference_location"
                ],
            )

            exact_reference_keys.add(
                key
            )

            severity, blocking = (
                classify_severity(
                    reference[
                        "reference_group"
                    ],
                    "exact_normalized",
                )
            )

            matches.append(
                make_match(
                    item,
                    reference,
                    "exact_normalized",
                    1.0,
                    1.0,
                    severity,
                    blocking,
                )
            )

        item_ngrams = (
            char_ngrams(
                normalized,
                ngram_size,
            )
        )

        for index, reference in enumerate(
            reference_items
        ):

            reference_key = (
                reference[
                    "reference_file"
                ],
                reference[
                    "reference_location"
                ],
            )

            if (
                reference_key
                in exact_reference_keys
            ):

                continue

            reference_normalized = (
                reference[
                    "normalized_text"
                ]
            )

            if substring_match(
                normalized,
                reference_normalized,
                substring_min_chars,
                substring_min_ratio,
            ):

                seq_score = (
                    sequence_similarity(
                        normalized,
                        reference_normalized,
                    )
                )

                jac_score = (
                    jaccard_similarity(
                        item_ngrams,
                        reference_ngrams[
                            index
                        ],
                    )
                )

                severity, blocking = (
                    classify_severity(
                        reference[
                            "reference_group"
                        ],
                        "substring",
                    )
                )

                matches.append(
                    make_match(
                        item,
                        reference,
                        "substring",
                        jac_score,
                        seq_score,
                        severity,
                        blocking,
                    )
                )

                continue

            jac_score = (
                jaccard_similarity(
                    item_ngrams,
                    reference_ngrams[
                        index
                    ],
                )
            )

            if (
                jac_score
                < jaccard_threshold
            ):

                continue

            seq_score = (
                sequence_similarity(
                    normalized,
                    reference_normalized,
                )
            )

            if (
                seq_score
                < sequence_threshold
            ):

                continue

            severity, blocking = (
                classify_severity(
                    reference[
                        "reference_group"
                    ],
                    "near_duplicate",
                )
            )

            matches.append(
                make_match(
                    item,
                    reference,
                    "near_duplicate",
                    jac_score,
                    seq_score,
                    severity,
                    blocking,
                )
            )

    return matches


def make_match(
    item: dict,
    reference: dict,
    match_type: str,
    jaccard_score: float,
    sequence_score: float,
    severity: str,
    blocking: bool,
) -> dict:

    return {
        "registry_item_id":
            item[
                "registry_item_id"
            ],

        "registry_item_type":
            item[
                "registry_item_type"
            ],

        "template_split":
            item[
                "template_split"
            ],

        "family":
            item[
                "family"
            ],

        "semantic_cluster_id":
            item[
                "semantic_cluster_id"
            ],

        "attack_objective":
            item[
                "attack_objective"
            ],

        "registry_variant":
            item[
                "variant"
            ],

        "registry_text":
            item[
                "text"
            ],

        "reference_group":
            reference[
                "reference_group"
            ],

        "reference_file":
            reference[
                "reference_file"
            ],

        "reference_location":
            reference[
                "reference_location"
            ],

        "reference_text":
            reference[
                "text"
            ],

        "match_type":
            match_type,

        "char_ngram_jaccard":
            round(
                jaccard_score,
                6,
            ),

        "sequence_similarity":
            round(
                sequence_score,
                6,
            ),

        "severity":
            severity,

        "blocking":
            blocking,
    }


def semantic_audit(
    registry_items: list[dict],
    reference_items: list[dict],
    model_name: str,
    threshold: float,
    batch_size: int,
    top_k: int,
) -> list[dict]:

    try:

        import numpy as np

        from sentence_transformers import (
            SentenceTransformer,
        )

    except ImportError as error:

        raise RuntimeError(
            "Semantic audit requires "
            "numpy and sentence-transformers"
        ) from error

    model = SentenceTransformer(
        model_name
    )

    registry_texts = [
        item[
            "text"
        ]
        for item
        in registry_items
    ]

    reference_texts = [
        item[
            "text"
        ]
        for item
        in reference_items
    ]

    registry_embeddings = (
        model.encode(
            registry_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )

    matches = []

    best = [
        []
        for _
        in registry_items
    ]

    chunk_size = 4096

    for start in range(
        0,
        len(
            reference_texts
        ),
        chunk_size,
    ):

        stop = min(
            start
            +
            chunk_size,
            len(
                reference_texts
            ),
        )

        chunk_embeddings = (
            model.encode(
                reference_texts[
                    start:
                    stop
                ],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        similarities = (
            np.matmul(
                registry_embeddings,
                chunk_embeddings.T,
            )
        )

        for registry_index in range(
            len(
                registry_items
            )
        ):

            row = similarities[
                registry_index
            ]

            candidates = (
                np.argpartition(
                    row,
                    -min(
                        top_k,
                        len(row),
                    ),
                )[
                    -min(
                        top_k,
                        len(row),
                    ):
                ]
            )

            for local_index in candidates:

                score = float(
                    row[
                        local_index
                    ]
                )

                if score < threshold:

                    continue

                reference_index = (
                    start
                    +
                    int(
                        local_index
                    )
                )

                best[
                    registry_index
                ].append(
                    (
                        score,
                        reference_index,
                    )
                )

    for registry_index, candidates in enumerate(
        best
    ):

        candidates = sorted(
            candidates,
            key=lambda value:
                value[0],
            reverse=True,
        )[
            :top_k
        ]

        item = registry_items[
            registry_index
        ]

        for score, reference_index in candidates:

            reference = (
                reference_items[
                    reference_index
                ]
            )

            severity, blocking = (
                classify_severity(
                    reference[
                        "reference_group"
                    ],
                    "semantic_similarity",
                )
            )

            match = (
                make_match(
                    item,
                    reference,
                    "semantic_similarity",
                    0.0,
                    0.0,
                    severity,
                    blocking,
                )
            )

            match[
                "semantic_similarity"
            ] = round(
                score,
                6,
            )

            match[
                "semantic_model"
            ] = model_name

            matches.append(
                match
            )

    return matches


def deduplicate_matches(
    matches: list[dict],
) -> list[dict]:

    priority = {
        "exact_normalized": 4,
        "substring": 3,
        "near_duplicate": 2,
        "semantic_similarity": 1,
    }

    result = {}

    for match in matches:

        key = (
            match[
                "registry_item_id"
            ],
            match[
                "registry_variant"
            ],
            match[
                "reference_group"
            ],
            match[
                "reference_file"
            ],
            match[
                "reference_location"
            ],
        )

        existing = result.get(
            key
        )

        if existing is None:

            result[
                key
            ] = match

            continue

        if (
            priority[
                match[
                    "match_type"
                ]
            ]
            >
            priority[
                existing[
                    "match_type"
                ]
            ]
        ):

            result[
                key
            ] = match

    return sorted(
        result.values(),
        key=lambda item: (
            not item[
                "blocking"
            ],
            item[
                "severity"
            ],
            item[
                "registry_item_id"
            ],
            item[
                "reference_group"
            ],
        ),
    )


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
        "--injecguard-clean",
        default=(
            "data/processed/defense/"
            "sources/injecguard_clean.jsonl"
        ),
    )

    parser.add_argument(
        "--injecguard-quarantine",
        default=(
            "data/processed/defense/"
            "sources/injecguard_quarantine.jsonl"
        ),
    )

    parser.add_argument(
        "--benchmark-dir",
        default=(
            "data/raw/"
            "defense_benchmarks"
        ),
    )

    parser.add_argument(
        "--ngram-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.72,
    )

    parser.add_argument(
        "--sequence-threshold",
        type=float,
        default=0.84,
    )

    parser.add_argument(
        "--substring-min-chars",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--substring-min-ratio",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--semantic-model",
        default=None,
    )

    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.88,
    )

    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--semantic-top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    registry_dir = Path(
        args.registry_dir
    )

    reference_paths = [
        Path(
            args.injecguard_clean
        ),
        Path(
            args.injecguard_quarantine
        ),
        Path(
            args.benchmark_dir
        ),
    ]

    print(
        "Loading registry..."
    )

    registry_items = (
        load_registry_items(
            registry_dir
        )
    )

    print(
        "Registry audit items:",
        len(
            registry_items
        ),
    )

    print(
        "Loading reference datasets..."
    )

    reference_items = (
        load_reference_items(
            reference_paths
        )
    )

    print(
        "Reference strings:",
        len(
            reference_items
        ),
    )

    reference_group_counts = Counter(
        item[
            "reference_group"
        ]
        for item
        in reference_items
    )

    print(
        "Running lexical contamination audit..."
    )

    matches = lexical_audit(
        registry_items,
        reference_items,
        args.ngram_size,
        args.jaccard_threshold,
        args.sequence_threshold,
        args.substring_min_chars,
        args.substring_min_ratio,
    )

    semantic_ran = False

    if args.semantic_model:

        print(
            "Running semantic contamination audit..."
        )

        semantic_matches = (
            semantic_audit(
                registry_items,
                reference_items,
                args.semantic_model,
                args.semantic_threshold,
                args.semantic_batch_size,
                args.semantic_top_k,
            )
        )

        matches.extend(
            semantic_matches
        )

        semantic_ran = True

    matches = deduplicate_matches(
        matches
    )

    blocking_matches = [
        match
        for match
        in matches
        if match[
            "blocking"
        ]
    ]

    match_type_counts = Counter(
        match[
            "match_type"
        ]
        for match
        in matches
    )

    severity_counts = Counter(
        match[
            "severity"
        ]
        for match
        in matches
    )

    source_counts = Counter(
        match[
            "reference_group"
        ]
        for match
        in matches
    )

    blocker_source_counts = Counter(
        match[
            "reference_group"
        ]
        for match
        in blocking_matches
    )

    status = (
        "PASS"
        if not blocking_matches
        else "FAIL"
    )

    report = {
        "status":
            status,

        "registry_version":
            "1.0",

        "registry_items_audited":
            len(
                registry_items
            ),

        "reference_strings_audited":
            len(
                reference_items
            ),

        "reference_groups":
            dict(
                sorted(
                    reference_group_counts.items()
                )
            ),

        "thresholds": {
            "char_ngram_size":
                args.ngram_size,

            "jaccard_threshold":
                args.jaccard_threshold,

            "sequence_threshold":
                args.sequence_threshold,

            "substring_min_chars":
                args.substring_min_chars,

            "substring_min_ratio":
                args.substring_min_ratio,

            "semantic_threshold":
                args.semantic_threshold,
        },

        "semantic_check": {
            "ran":
                semantic_ran,

            "model":
                args.semantic_model,
        },

        "matches":
            len(
                matches
            ),

        "blocking_matches":
            len(
                blocking_matches
            ),

        "match_types":
            dict(
                sorted(
                    match_type_counts.items()
                )
            ),

        "severity":
            dict(
                sorted(
                    severity_counts.items()
                )
            ),

        "matches_by_reference_group":
            dict(
                sorted(
                    source_counts.items()
                )
            ),

        "blockers_by_reference_group":
            dict(
                sorted(
                    blocker_source_counts.items()
                )
            ),

        "policy": {
            "bipia_exact_or_substring":
                "block",

            "bipia_near_duplicate":
                "block",

            "bipia_semantic_similarity":
                "manual_review",

            "notinject_overlap":
                "overdefense_review",

            "injecguard_overlap":
                "report_only",

            "semantic_similarity":
                "manual_review",
        },
    }

    report_path = (
        registry_dir
        /
        "contamination_report.json"
    )

    pairs_path = (
        registry_dir
        /
        "contamination_pairs.jsonl"
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
        matches,
        pairs_path,
    )

    print(
        "\n================================"
    )

    print(
        "Template Registry Contamination Audit"
    )

    print(
        "================================"
    )

    print(
        "\nRegistry audit items:",
        len(
            registry_items
        ),
    )

    print(
        "Reference strings:",
        len(
            reference_items
        ),
    )

    print(
        "\nMatches:",
        len(
            matches
        ),
    )

    print(
        "Blocking matches:",
        len(
            blocking_matches
        ),
    )

    print(
        "\nMatches by type:"
    )

    for name, count in sorted(
        match_type_counts.items()
    ):

        print(
            f"  {name}: {count}"
        )

    print(
        "\nMatches by source:"
    )

    for name, count in sorted(
        source_counts.items()
    ):

        print(
            f"  {name}: {count}"
        )

    print(
        "\nSemantic check:",
        (
            "enabled"
            if semantic_ran
            else "not run"
        ),
    )

    print(
        "Status:",
        status,
    )

    print(
        "\nReport:",
        report_path,
    )

    print(
        "Pairs:",
        pairs_path,
    )

    if blocking_matches:

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()