from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

FEVER_URLS = {
    "train":
        "https://fever.ai/download/fever/train.jsonl",

    "paper_dev":
        "https://fever.ai/download/fever/paper_dev.jsonl",

    "paper_test":
        "https://fever.ai/download/fever/paper_test.jsonl",
}

LABEL_MAP = {
    "SUPPORTS": "True",
    "REFUTES": "False",
}

def load_jsonl(path: Path) -> list[dict]:

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
                    f"Invalid JSON in {path} "
                    f"at line {line_number}"
                ) from error

            records.append(record)

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

def download_file(
    url: str,
    destination: Path,
) -> None:

    if destination.exists():

        print(
            f"Already downloaded: "
            f"{destination}"
        )

        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Downloading:\n"
        f"  {url}"
    )

    with requests.get(
        url,
        stream=True,
        timeout=120,
    ) as response:

        response.raise_for_status()

        with destination.open(
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    file.write(chunk)

    print(
        f"Saved: {destination}"
    )

def convert_evidence_item(
    item,
) -> dict:

    if (
        not isinstance(item, list)
        or len(item) != 4
    ):

        raise ValueError(
            f"Invalid evidence item: {item}"
        )

    (
        annotation_id,
        evidence_id,
        page,
        sentence_id,
    ) = item

    if not isinstance(page, str):
        raise ValueError(
            f"Invalid evidence page: {page!r}"
        )

    if not isinstance(sentence_id, int):
        raise ValueError(
            f"Invalid sentence_id: "
            f"{sentence_id!r}"
        )

    return {
        "annotation_id":
            annotation_id,

        "evidence_id":
            evidence_id,

        "page":
            page,

        "sentence_id":
            sentence_id,
    }

def convert_evidence_sets(
    raw_evidence,
) -> list[list[dict]]:

    if not isinstance(
        raw_evidence,
        list,
    ):

        raise ValueError(
            "FEVER evidence must be a list."
        )

    evidence_sets = []

    for raw_set in raw_evidence:

        if not isinstance(
            raw_set,
            list,
        ):

            raise ValueError(
                f"Invalid evidence set: "
                f"{raw_set!r}"
            )

        evidence_set = []

        for item in raw_set:

            evidence_set.append(
                convert_evidence_item(
                    item
                )
            )

        if evidence_set:
            evidence_sets.append(
                evidence_set
            )

    return evidence_sets

def reconstruct_split(
    raw_path: Path,
    clean_audit_path: Path,
    split_name: str,
) -> tuple[list[dict], dict]:

    clean_records = load_jsonl(
        clean_audit_path
    )

    clean_by_id = {
        int(record["claim_id"]):
            record
        for record in clean_records
    }

    target_ids = set(
        clean_by_id
    )
    reconstructed = []
    found_ids = set()

    with raw_path.open(
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

            raw = json.loads(line)

            claim_id = int(
                raw["id"]
            )

            if claim_id not in target_ids:
                continue

            audit = clean_by_id[
                claim_id
            ]

            if raw["claim"] != audit["claim"]:

                raise ValueError(
                    f"{split_name}: "
                    f"claim mismatch "
                    f"for claim_id="
                    f"{claim_id}"
                )

            fever_label = raw["label"]

            if fever_label not in LABEL_MAP:
                raise ValueError(
                    f"{split_name}: "
                    f"unexpected label "
                    f"{fever_label!r} "
                    f"for claim_id="
                    f"{claim_id}"
                )

            expected_label = (
                LABEL_MAP[
                    fever_label
                ]
            )

            if (
                expected_label
                != audit["label"]
            ):
                raise ValueError(
                    f"{split_name}: "
                    f"label mismatch "
                    f"for claim_id="
                    f"{claim_id}"
                )

            evidence_sets = (
                convert_evidence_sets(
                    raw.get(
                        "evidence",
                        [],
                    )
                )
            )

            if not evidence_sets:
                raise ValueError(
                    f"{split_name}: "
                    f"no gold evidence "
                    f"for binary claim "
                    f"{claim_id}"
                )

            reconstructed.append(
                {
                    "claim_id":
                        claim_id,
                    "claim":
                        audit["claim"],
                    "label":
                        audit["label"],
                    "fever_label":
                        fever_label,
                    "evidence_sets":
                        evidence_sets,
                    "split":
                        split_name,
                }
            )
            found_ids.add(
                claim_id
            )

    missing_ids = (
        target_ids
        -
        found_ids
    )

    if missing_ids:
        examples = sorted(
            missing_ids
        )[:10]
        raise ValueError(
            f"{split_name}: "
            f"{len(missing_ids)} "
            f"clean claims were not "
            f"found in original FEVER. "
            f"Examples: {examples}"
        )

    reconstructed.sort(
        key=lambda item:
            item["claim_id"]
    )

    total_sets = sum(
        len(record["evidence_sets"])
        for record in reconstructed
    )

    total_sentences = sum(
        len(evidence_set)
        for record in reconstructed
        for evidence_set
        in record["evidence_sets"]
    )

    multi_set_claims = sum(
        len(
            record["evidence_sets"]
        ) > 1
        for record in reconstructed
    )

    multi_sentence_sets = sum(
        len(evidence_set) > 1
        for record in reconstructed
        for evidence_set
        in record["evidence_sets"]
    )

    stats = {
        "claims":
            len(reconstructed),
        "evidence_sets":
            total_sets,
        "evidence_sentences":
            total_sentences,
        "claims_with_multiple_sets":
            multi_set_claims,
        "multi_sentence_sets":
            multi_sentence_sets,
    }

    return (reconstructed, stats)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-audit-dir",
        default=(
            "data/processed/"
            "fever/clean/audit"
        ),
    )

    parser.add_argument(
        "--raw-dir",
        default=(
            "data/raw/fever/original"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/"
            "fever/gold/references"
        ),
    )

    args = parser.parse_args()
    clean_audit_dir = Path(args.clean_audit_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    report = {}

    for split_name, url \
            in FEVER_URLS.items():

        print(
            f"\nProcessing "
            f"{split_name}..."
        )
        raw_path = (raw_dir/f"{split_name}.jsonl")
        download_file(url, raw_path)

        (
            records,
            stats,
        ) = reconstruct_split(
            raw_path=raw_path,
            clean_audit_path=(clean_audit_dir/f"{split_name}.jsonl"),
            split_name=split_name,
        )

        output_path = (output_dir/f"{split_name}.jsonl")

        write_jsonl(
            records,
            output_path,
        )

        report[split_name] = stats
        print(
            f"Claims: "
            f"{stats['claims']}"
        )
        print(
            f"Evidence sets: "
            f"{stats['evidence_sets']}"
        )
        print(
            f"Evidence sentences: "
            f"{stats['evidence_sentences']}"
        )
        print("Claims with multiple evidence sets:", stats["claims_with_multiple_sets"])
        print("Multi-sentence evidence sets:", stats["multi_sentence_sets"])

    report_path = (output_dir.parent/"reference_report.json")
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nGold evidence references "
        "reconstructed successfully."
    )

    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()