from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import requests

BENCHMARK_URLS = {
    "notinject_one": (
        "https://raw.githubusercontent.com/"
        "leolee99/PIGuard/main/datasets/NotInject_one.json"
    ),
    "notinject_two": (
        "https://raw.githubusercontent.com/"
        "leolee99/PIGuard/main/datasets/NotInject_two.json"
    ),
    "notinject_three": (
        "https://raw.githubusercontent.com/"
        "leolee99/PIGuard/main/datasets/NotInject_three.json"
    ),
    "piguard_bipia_text": (
        "https://raw.githubusercontent.com/"
        "leolee99/PIGuard/main/datasets/BIPIA_text.json"
    ),
    "piguard_bipia_code": (
        "https://raw.githubusercontent.com/"
        "leolee99/PIGuard/main/datasets/BIPIA_code.json"
    ),
    "bipia_text_attack_train": (
        "https://raw.githubusercontent.com/"
        "microsoft/BIPIA/main/benchmark/text_attack_train.json"
    ),
    "bipia_text_attack_test": (
        "https://raw.githubusercontent.com/"
        "microsoft/BIPIA/main/benchmark/text_attack_test.json"
    ),
    "bipia_code_attack_train": (
        "https://raw.githubusercontent.com/"
        "microsoft/BIPIA/main/benchmark/code_attack_train.json"
    ),
    "bipia_code_attack_test": (
        "https://raw.githubusercontent.com/"
        "microsoft/BIPIA/main/benchmark/code_attack_test.json"
    ),
}

def normalize_text(text: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = " ".join(
        text.strip().split()
    )

    return text.casefold()

def text_sha256(text: str) -> str:
    return hashlib.sha256(
        normalize_text(text).encode(
            "utf-8"
        )
    ).hexdigest()

def download_json(
    name: str,
    url: str,
    directory: Path,
) -> Path:

    path = directory / f"{name}.json"

    if path.exists():
        print(f"Already exists: {path}")
        return path

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Downloading {name}...")

    response = requests.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    path.write_bytes(
        response.content
    )
    print(f"Saved: {path}")

    return path

def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def load_train(path: Path) -> list[dict]:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(
            "InjecGuard train.json must be a list."
        )

    return data

def extract_strings(value) -> list[str]:
    result = []

    if isinstance(value, str):
        if value.strip():
            result.append(value)

    elif isinstance(value, list):
        for item in value:
            result.extend(
                extract_strings(item)
            )

    elif isinstance(value, dict):
        for item in value.values():
            result.extend(
                extract_strings(item)
            )

    return result

def extract_notinject_prompts(
    data,
) -> list[str]:
    result = []

    if not isinstance(data, list):
        return result

    for item in data:
        if not isinstance(item, dict):
            continue

        prompt = item.get(
            "prompt"
        )

        if (
            isinstance(prompt, str)
            and prompt.strip()
        ):
            result.append(prompt)

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

def build_benchmark_sets(
    benchmark_paths: dict[str, Path],
):
    notinject_texts = []

    for name in (
        "notinject_one",
        "notinject_two",
        "notinject_three",
    ):
        data = load_json(
            benchmark_paths[name]
        )

        notinject_texts.extend(
            extract_notinject_prompts(
                data
            )
        )

    bipia_full_texts = []

    for name in (
        "piguard_bipia_text",
        "piguard_bipia_code",
    ):
        data = load_json(
            benchmark_paths[name]
        )

        bipia_full_texts.extend(
            extract_strings(
                data
            )
        )

    bipia_train_payloads = []
    bipia_test_payloads = []

    for name in (
        "bipia_text_attack_train",
        "bipia_code_attack_train",
    ):
        data = load_json(
            benchmark_paths[name]
        )

        bipia_train_payloads.extend(
            extract_strings(
                data
            )
        )

    for name in (
        "bipia_text_attack_test",
        "bipia_code_attack_test",
    ):
        data = load_json(
            benchmark_paths[name]
        )

        bipia_test_payloads.extend(
            extract_strings(
                data
            )
        )

    return {
        "notinject":
            sorted(set(
                notinject_texts
            )),
        "bipia_full":
            sorted(set(
                bipia_full_texts
            )),
        "bipia_train_payload":
            sorted(set(
                bipia_train_payloads
            )),
        "bipia_test_payload":
            sorted(set(
                bipia_test_payloads
            )),
    }

def build_hash_set(
    texts: list[str],
) -> set[str]:
    return {
        text_sha256(text)
        for text in texts
        if text.strip()
    }

def prepare_payloads(
    texts: list[str],
    min_length: int,
):
    payloads = []

    seen = set()

    for text in texts:
        normalized = normalize_text(
            text
        )

        if len(normalized) < min_length:
            continue

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        payloads.append(
            normalized
        )

    payloads.sort(
        key=len,
        reverse=True,
    )

    return payloads

def find_payload_matches(
    normalized_prompt: str,
    payloads: list[str],
    max_matches: int = 5,
):
    matches = []

    for payload in payloads:
        if payload in normalized_prompt:
            matches.append(
                payload
            )

            if len(matches) >= max_matches:
                break

    return matches

def audit_contamination(
    records: list[dict],
    benchmark_sets: dict,
    min_payload_length: int,
):
    notinject_hashes = build_hash_set(
        benchmark_sets[
            "notinject"
        ]
    )

    bipia_full_hashes = build_hash_set(
        benchmark_sets[
            "bipia_full"
        ]
    )

    train_payloads = prepare_payloads(
        benchmark_sets[
            "bipia_train_payload"
        ],
        min_payload_length,
    )

    test_payloads = prepare_payloads(
        benchmark_sets[
            "bipia_test_payload"
        ],
        min_payload_length,
    )

    contaminated = []

    source_stats = defaultdict(
        Counter
    )

    for index, record in enumerate(records):
        prompt = record.get(
            "prompt"
        )

        if (
            not isinstance(prompt, str)
            or
            not prompt.strip()
        ):
            continue

        source = str(
            record.get(
                "source",
                "<missing>",
            )
        )

        label = record.get(
            "label"
        )

        normalized = normalize_text(
            prompt
        )

        digest = text_sha256(
            prompt
        )

        reasons = []

        notinject_match = (
            digest in notinject_hashes
        )

        bipia_full_match = (
            digest in bipia_full_hashes
        )

        train_payload_matches = (
            find_payload_matches(
                normalized,
                train_payloads,
            )
        )

        test_payload_matches = (
            find_payload_matches(
                normalized,
                test_payloads,
            )
        )

        if notinject_match:
            reasons.append(
                "notinject_full_text"
            )

        if bipia_full_match:
            reasons.append(
                "bipia_full_text"
            )

        if train_payload_matches:
            reasons.append(
                "bipia_train_payload"
            )

        if test_payload_matches:
            reasons.append(
                "bipia_test_payload"
            )

        source_stats[
            source
        ][
            "total"
        ] += 1

        for reason in reasons:
            source_stats[
                source
            ][
                reason
            ] += 1

        if reasons:

            contaminated.append(
                {
                    "index":
                        index,

                    "source":
                        source,

                    "label":
                        label,

                    "reasons":
                        reasons,

                    "text_sha256":
                        digest,

                    "prompt":
                        prompt,

                    "bipia_train_payload_matches":
                        train_payload_matches,

                    "bipia_test_payload_matches":
                        test_payload_matches,
                }
            )

    report_sources = []

    for source in sorted(
        source_stats
    ):

        stats = source_stats[
            source
        ]

        contaminated_count = sum(
            1
            for item in contaminated
            if item[
                "source"
            ] == source
        )

        report_sources.append(
            {
                "source":
                    source,

                "total":
                    stats[
                        "total"
                    ],

                "contaminated_samples":
                    contaminated_count,

                "notinject_full_text":
                    stats[
                        "notinject_full_text"
                    ],

                "bipia_full_text":
                    stats[
                        "bipia_full_text"
                    ],

                "bipia_train_payload":
                    stats[
                        "bipia_train_payload"
                    ],

                "bipia_test_payload":
                    stats[
                        "bipia_test_payload"
                    ],
            }
        )

    summary = {
        "total_train_records":
            len(records),

        "contaminated_samples":
            len(
                contaminated
            ),

        "notinject_full_text_matches":
            sum(
                "notinject_full_text"
                in item["reasons"]
                for item in contaminated
            ),

        "bipia_full_text_matches":
            sum(
                "bipia_full_text"
                in item["reasons"]
                for item in contaminated
            ),

        "bipia_train_payload_matches":
            sum(
                "bipia_train_payload"
                in item["reasons"]
                for item in contaminated
            ),

        "bipia_test_payload_matches":
            sum(
                "bipia_test_payload"
                in item["reasons"]
                for item in contaminated
            ),

        "benchmark_sizes": {
            key:
                len(value)
            for key, value
            in benchmark_sets.items()
        },

        "min_payload_length":
            min_payload_length,
    }

    return (
        summary,
        report_sources,
        contaminated,
    )


def print_report(
    summary: dict,
    sources: list[dict],
):

    print(
        "\n================================"
    )

    print(
        "Benchmark Contamination Audit"
    )

    print(
        "================================"
    )

    print(
        "\nTotal train records:",
        summary[
            "total_train_records"
        ],
    )

    print(
        "Contaminated samples:",
        summary[
            "contaminated_samples"
        ],
    )

    print(
        "\nNotInject full-text matches:",
        summary[
            "notinject_full_text_matches"
        ],
    )

    print(
        "BIPIA full-text matches:",
        summary[
            "bipia_full_text_matches"
        ],
    )

    print(
        "BIPIA train-payload matches:",
        summary[
            "bipia_train_payload_matches"
        ],
    )

    print(
        "BIPIA test-payload matches:",
        summary[
            "bipia_test_payload_matches"
        ],
    )

    print(
        "\nSources with contamination:"
    )

    for item in sources:

        if (
            item[
                "contaminated_samples"
            ] == 0
        ):
            continue

        print(
            f"\n{item['source']}"
        )

        print(
            "  contaminated:",
            item[
                "contaminated_samples"
            ],
        )

        print(
            "  NotInject:",
            item[
                "notinject_full_text"
            ],
        )

        print(
            "  BIPIA full:",
            item[
                "bipia_full_text"
            ],
        )

        print(
            "  BIPIA train payload:",
            item[
                "bipia_train_payload"
            ],
        )

        print(
            "  BIPIA test payload:",
            item[
                "bipia_test_payload"
            ],
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default=(
            "data/raw/injecguard/"
            "train.json"
        ),
    )

    parser.add_argument(
        "--benchmark-dir",
        default=(
            "data/raw/defense_benchmarks"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/defense/"
            "audit/injecguard"
        ),
    )

    parser.add_argument(
        "--min-payload-length",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    train_path = Path(
        args.train
    )

    benchmark_dir = Path(
        args.benchmark_dir
    )

    output_dir = Path(
        args.output_dir
    )

    benchmark_paths = {}

    for name, url in (
        BENCHMARK_URLS.items()
    ):

        benchmark_paths[
            name
        ] = download_json(
            name,
            url,
            benchmark_dir,
        )

    records = load_train(
        train_path
    )

    benchmark_sets = (
        build_benchmark_sets(
            benchmark_paths
        )
    )

    (
        summary,
        source_report,
        contaminated,

    ) = audit_contamination(
        records,
        benchmark_sets,
        args.min_payload_length,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        /
        "contamination_audit.json"
    )

    report_path.write_text(
        json.dumps(
            {
                "summary":
                    summary,

                "sources":
                    source_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_jsonl(
        contaminated,
        output_dir
        /
        "contaminated_samples.jsonl",
    )

    print_report(
        summary,
        source_report,
    )

    print(
        "\nReport:",
        report_path,
    )

if __name__ == "__main__":
    main()