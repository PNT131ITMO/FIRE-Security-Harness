from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import defaultdict, Counter
from pathlib import Path


SPLITS = (
    "train",
    "paper_dev",
    "paper_test",
)


def load_jsonl(path: Path) -> list[dict]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

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


def extract_sentence_ids(
    lines_text: str,
) -> set[int]:

    ids = set()

    for raw_line in lines_text.splitlines():

        if not raw_line:
            continue

        parts = raw_line.split(
            "\t",
            1,
        )

        if not parts:
            continue

        try:

            sentence_id = int(
                parts[0]
            )

        except ValueError:

            continue

        ids.add(
            sentence_id
        )

    return ids


def inspect_wiki(
    wiki_zip: Path,
    target_pages: set[str],
):

    page_sentence_ids = {}

    print(
        "\nScanning target pages "
        "inside wiki dump..."
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

        for index, member in enumerate(
            members,
            start=1,
        ):

            print(
                f"\rScanning "
                f"{index}/{len(members)}",
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

                    page = json.loads(
                        line
                    )

                    page_id = page.get(
                        "id"
                    )

                    if page_id not in target_pages:
                        continue

                    page_sentence_ids[
                        page_id
                    ] = extract_sentence_ids(
                        page.get(
                            "lines",
                            "",
                        )
                    )

    print()

    return page_sentence_ids


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--unresolved",
        default=(
            "data/processed/fever/"
            "gold/unresolved_evidence.jsonl"
        ),
    )

    parser.add_argument(
        "--reference-dir",
        default=(
            "data/processed/fever/"
            "gold/references"
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
            "data/processed/fever/"
            "gold/diagnostics"
        ),
    )

    args = parser.parse_args()

    unresolved_path = Path(
        args.unresolved
    )

    reference_dir = Path(
        args.reference_dir
    )

    wiki_zip = Path(
        args.wiki_zip
    )

    output_dir = Path(
        args.output_dir
    )

    unresolved = load_jsonl(
        unresolved_path
    )

    print(
        "Unresolved occurrences:",
        len(unresolved),
    )

    unique_refs = {
        (
            item["page"],
            int(item["sentence_id"]),
        )
        for item in unresolved
    }

    target_pages = {
        page
        for page, _
        in unique_refs
    }

    print(
        "Unique unresolved references:",
        len(unique_refs),
    )

    print(
        "Unique affected pages:",
        len(target_pages),
    )

    page_sentence_ids = inspect_wiki(
        wiki_zip,
        target_pages,
    )

    diagnostic_records = []

    reason_counter = Counter()

    for page, sentence_id in sorted(
        unique_refs
    ):

        if page not in page_sentence_ids:

            reason = (
                "page_not_found"
            )

        elif (
            sentence_id
            not in
            page_sentence_ids[page]
        ):

            reason = (
                "sentence_id_missing"
            )

        else:

            reason = (
                "resolver_logic_issue"
            )

        reason_counter[
            reason
        ] += 1

        diagnostic_records.append(
            {
                "page":
                    page,

                "sentence_id":
                    sentence_id,

                "reason":
                    reason,
            }
        )

    bad_sets = defaultdict(set)

    for item in unresolved:

        key = (
            item["split"],
            int(item["claim_id"]),
        )

        bad_sets[key].add(
            int(
                item[
                    "evidence_set_index"
                ]
            )
        )

    split_reports = {}

    affected_claim_records = []

    for split in SPLITS:

        references = load_jsonl(
            reference_dir
            /
            f"{split}.jsonl"
        )

        touched = 0
        no_complete_set = 0
        still_has_complete_set = 0

        for record in references:

            claim_id = int(
                record["claim_id"]
            )

            key = (
                split,
                claim_id,
            )

            if key not in bad_sets:
                continue

            touched += 1

            total_sets = len(
                record["evidence_sets"]
            )

            invalid_sets = (
                bad_sets[key]
            )

            complete_sets = (
                total_sets
                -
                len(invalid_sets)
            )

            if complete_sets > 0:

                still_has_complete_set += 1

                status = (
                    "has_complete_evidence_set"
                )

            else:

                no_complete_set += 1

                status = (
                    "no_complete_evidence_set"
                )

            affected_claim_records.append(
                {
                    "split":
                        split,

                    "claim_id":
                        claim_id,

                    "claim":
                        record["claim"],

                    "total_evidence_sets":
                        total_sets,

                    "invalid_evidence_sets":
                        sorted(
                            invalid_sets
                        ),

                    "complete_evidence_sets":
                        complete_sets,

                    "status":
                        status,
                }
            )

        split_reports[
            split
        ] = {
            "total_claims":
                len(references),

            "claims_touched_by_unresolved":
                touched,

            "claims_still_with_complete_set":
                still_has_complete_set,

            "claims_without_complete_set":
                no_complete_set,

            "usable_claims":
                len(references)
                -
                no_complete_set,
        }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        diagnostic_records,

        output_dir
        /
        "unresolved_reference_reasons.jsonl",
    )

    write_jsonl(
        affected_claim_records,

        output_dir
        /
        "affected_claims.jsonl",
    )

    report = {
        "unresolved_occurrences":
            len(unresolved),

        "unique_unresolved_references":
            len(unique_refs),

        "unique_affected_pages":
            len(target_pages),

        "reason_counts":
            dict(reason_counter),

        "splits":
            split_reports,
    }

    report_path = (
        output_dir
        /
        "diagnostic_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nReference-level diagnosis"
    )

    for reason, count \
            in reason_counter.items():

        print(
            f"{reason}: {count}"
        )

    print(
        "\nClaim-level impact"
    )

    for split in SPLITS:

        stats = split_reports[
            split
        ]

        print(
            f"\n{split}"
        )

        print(
            "  total claims:",
            stats["total_claims"],
        )

        print(
            "  touched by unresolved:",
            stats[
                "claims_touched_by_unresolved"
            ],
        )

        print(
            "  still have complete evidence set:",
            stats[
                "claims_still_with_complete_set"
            ],
        )

        print(
            "  no complete evidence set:",
            stats[
                "claims_without_complete_set"
            ],
        )

        print(
            "  usable claims:",
            stats[
                "usable_claims"
            ],
        )

    print(
        "\nReport:",
        report_path,
    )
if __name__ == "__main__":
    main()