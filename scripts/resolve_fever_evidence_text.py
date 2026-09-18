from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
import re
from collections import defaultdict
from pathlib import Path

import requests

SPLITS = (
    "train",
    "paper_dev",
    "paper_test",
)

WIKI_URL = (
    "https://fever.ai/download/fever/"
    "wiki-pages.zip"
)

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

def download_wiki_zip(
    url: str,
    destination: Path,
) -> None:
    if destination.exists():
        print(
            f"Wikipedia dump already exists: "
            f"{destination}"
        )
        return
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp_path = destination.with_suffix(
        destination.suffix + ".part"
    )
    downloaded = (
        temp_path.stat().st_size
        if temp_path.exists()
        else 0
    )
    headers = {}
    if downloaded > 0:
        headers["Range"] = (
            f"bytes={downloaded}-"
        )
        print(
            "Resuming download from "
            f"{downloaded / 1024**3:.2f} GB"
        )
    else:
        print(
            "Downloading FEVER Wikipedia dump..."
        )

    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    if (
        downloaded > 0
        and response.status_code == 206
    ):

        mode = "ab"

        total = (
            downloaded
            +
            int(
                response.headers.get(
                    "Content-Length",
                    0,
                )
            )
        )
    else:
        if downloaded > 0:
            print(
                "Server does not support resume. "
                "Restarting download."
            )
        mode = "wb"
        downloaded = 0

        total = int(
            response.headers.get(
                "Content-Length",
                0,
            )
        )

    last_reported_mb = -1

    with temp_path.open(mode) as file:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if not chunk:
                continue
            file.write(chunk)
            downloaded += len(chunk)
            downloaded_mb = (
                downloaded // (100 * 1024 * 1024)
            )

            if (
                downloaded_mb
                != last_reported_mb
            ):
                last_reported_mb = (
                    downloaded_mb
                )
                if total:
                    print(
                        "\rDownloaded: "
                        f"{downloaded / 1024**3:.2f} / "
                        f"{total / 1024**3:.2f} GB",
                        end="",
                    )
                else:
                    print(
                        "\rDownloaded: "
                        f"{downloaded / 1024**3:.2f} GB",
                        end="",
                    )
    print()

    os.replace(
        temp_path,
        destination,
    )
    print(
        f"Saved: {destination}"
    )

def collect_required_references(
    reference_dir: Path,
):

    records_by_split = {}
    required = defaultdict(set)
    total_reference_occurrences = 0
    for split_name in SPLITS:
        path = (
            reference_dir
            /
            f"{split_name}.jsonl"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing Step 2A output: {path}"
            )

        records = load_jsonl(
            path
        )
        records_by_split[
            split_name
        ] = records

        for record in records:
            for evidence_set \
                    in record["evidence_sets"]:
                for evidence in evidence_set:
                    page = evidence["page"]

                    sentence_id = int(
                        evidence["sentence_id"]
                    )

                    required[
                        page
                    ].add(
                        sentence_id
                    )

                    total_reference_occurrences += 1

    unique_sentence_refs = sum(
        len(sentence_ids)
        for sentence_ids
        in required.values()
    )

    print(
        "\nRequired evidence:"
    )

    print(
        "  unique Wikipedia pages:",
        len(required),
    )

    print(
        "  unique sentence references:",
        unique_sentence_refs,
    )

    print(
        "  total evidence occurrences:",
        total_reference_occurrences,
    )

    return (
        records_by_split,
        required,
    )

def extract_sentences(
    lines_text: str,
    wanted_ids: set[int],
) -> dict[int, str]:

    found = {}
    for raw_line in lines_text.splitlines():

        if not raw_line:
            continue
        parts = raw_line.split("\t")
        if len(parts) < 2:
            continue
        try:

            sentence_id = int(
                parts[0]
            )
        except ValueError:
            continue
        if sentence_id not in wanted_ids:
            continue

        sentence_text = parts[1]

        found[
            sentence_id
        ] = sentence_text

    return found

def resolve_references(
    wiki_zip: Path,
    required: dict[str, set[int]],
):

    resolved = {}

    missing_pages = set(
        required.keys()
    )

    total_required_sentences = sum(
        len(ids)
        for ids in required.values()
    )

    resolved_sentence_count = 0

    print(
        "\nScanning Wikipedia archive..."
    )

    with zipfile.ZipFile(
        wiki_zip,
        "r",
    ) as archive:
        members = [
            name
            for name in archive.namelist()
            
            if re.fullmatch(
                r"(?:.*/)?wiki-\d{3}\.jsonl",
                name,
            )
        ]

        members.sort()
        print(
            "Wikipedia JSONL files:",
            len(members),
        )

        for file_index, member in enumerate(
            members,
            start=1,
        ):

            print(
                f"\rScanning "
                f"{file_index}/{len(members)}: "
                f"{Path(member).name}",
                end="",
            )

            with archive.open(
                member,
                "r",
            ) as raw_file:

                text_file = io.TextIOWrapper(
                    raw_file,
                    encoding="utf-8",
                )

                for line in text_file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        page = json.loads(
                            line
                        )
                    except json.JSONDecodeError:
                        continue

                    page_id = page.get(
                        "id"
                    )

                    if page_id not in required:
                        continue

                    wanted_ids = required[
                        page_id
                    ]
                    sentences = extract_sentences(
                        page.get(
                            "lines",
                            "",
                        ),
                        wanted_ids,
                    )

                    resolved[
                        page_id
                    ] = sentences

                    resolved_sentence_count += (
                        len(sentences)
                    )

                    missing_pages.discard(
                        page_id
                    )

            if not missing_pages:
                break

    print()

    print(
        "Resolved sentence references:",
        f"{resolved_sentence_count}"
        f"/{total_required_sentences}",
    )

    return resolved

def enrich_records(
    records: list[dict],
    resolved: dict,
):
    output_records = []
    unresolved = []
    empty_sentences = []

    for record in records:
        new_record = dict(
            record
        )
        new_sets = []
        for set_index, evidence_set \
                in enumerate(
                    record["evidence_sets"]
                ):
            new_set = []
            for evidence_index, evidence \
                    in enumerate(
                        evidence_set
                    ):
                page = evidence[
                    "page"
                ]
                sentence_id = int(
                    evidence[
                        "sentence_id"
                    ]
                )
                sentence_text = (
                    resolved
                    .get(
                        page,
                        {},
                    )
                    .get(
                        sentence_id
                    )
                )

                if sentence_text is None:

                    unresolved.append(
                        {
                            "claim_id":
                                record[
                                    "claim_id"
                                ],

                            "page":
                                page,

                            "sentence_id":
                                sentence_id,

                            "evidence_set_index":
                                set_index,

                            "evidence_index":
                                evidence_index,
                        }
                    )
                    continue

                if not sentence_text.strip():

                    empty_sentences.append(
                        {
                            "claim_id":
                                record[
                                    "claim_id"
                                ],

                            "page":
                                page,

                            "sentence_id":
                                sentence_id,
                        }
                    )

                new_evidence = dict(
                    evidence
                )

                new_evidence[
                    "text"
                ] = sentence_text

                new_set.append(
                    new_evidence
                )

            new_sets.append(
                new_set
            )

        new_record[
            "evidence_sets"
        ] = new_sets

        output_records.append(
            new_record
        )

    return (
        output_records,
        unresolved,
        empty_sentences,
    )

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference-dir",
        default=(
            "data/processed/"
            "fever/gold/references"
        ),
    )

    parser.add_argument(
        "--wiki-zip",
        default=(
            "data/raw/fever/"
            "wiki-pages.zip"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/"
            "fever/gold/resolved"
        ),
    )

    args = parser.parse_args()

    reference_dir = Path(
        args.reference_dir
    )

    wiki_zip = Path(
        args.wiki_zip
    )

    output_dir = Path(
        args.output_dir
    )

    (
        records_by_split,
        required,

    ) = collect_required_references(
        reference_dir
    )

    download_wiki_zip(
        WIKI_URL,
        wiki_zip,
    )

    resolved = resolve_references(
        wiki_zip,
        required,
    )

    report = {}

    all_unresolved = []

    all_empty = []

    for split_name in SPLITS:

        print(
            f"\nBuilding resolved "
            f"{split_name}..."
        )

        (
            resolved_records,
            unresolved,
            empty_sentences,

        ) = enrich_records(

            records_by_split[
                split_name
            ],

            resolved,
        )

        write_jsonl(
            resolved_records,

            output_dir
            /
            f"{split_name}.jsonl",
        )

        for item in unresolved:

            item["split"] = (
                split_name
            )

        for item in empty_sentences:

            item["split"] = (
                split_name
            )

        all_unresolved.extend(
            unresolved
        )

        all_empty.extend(
            empty_sentences
        )

        report[
            split_name
        ] = {

            "claims":
                len(
                    resolved_records
                ),

            "unresolved_evidence":
                len(
                    unresolved
                ),

            "empty_evidence_text":
                len(
                    empty_sentences
                ),
        }

        print(
            "Claims:",
            len(
                resolved_records
            ),
        )

        print(
            "Unresolved evidence:",
            len(
                unresolved
            ),
        )

        print(
            "Empty evidence text:",
            len(
                empty_sentences
            ),
        )

    write_jsonl(
        all_unresolved,
        output_dir.parent
        /
        "unresolved_evidence.jsonl",
    )

    write_jsonl(
        all_empty,
        output_dir.parent
        /
        "empty_evidence.jsonl",
    )

    report[
        "total_unresolved_evidence"
    ] = len(
        all_unresolved
    )

    report[
        "total_empty_evidence"
    ] = len(
        all_empty
    )

    report_path = (
        output_dir.parent
        /
        "resolution_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if all_unresolved:

        raise RuntimeError(
            f"Gold evidence reconstruction "
            f"is incomplete: "
            f"{len(all_unresolved)} "
            f"evidence references "
            f"could not be resolved. "
            f"See "
            f"{output_dir.parent / 'unresolved_evidence.jsonl'}"
        )

    print(
        "\nGold evidence text "
        "resolved successfully."
    )

    print(
        "Output:",
        output_dir,
    )

    print(
        "Report:",
        report_path,
    )

if __name__ == "__main__":
    main()