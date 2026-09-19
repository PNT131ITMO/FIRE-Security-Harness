from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_POLICY = {
    "BIPIA": {
        "semantic_type": "indirect_prompt_injection",
        "decision": "remove",
        "contamination_risk": "critical",
        "reason": "Reserved final BIPIA benchmark.",
    },

    "TaskTracker": {
        "semantic_type": "indirect_task_drift",
        "decision": "review",
        "contamination_risk": "high",
        "reason": (
            "Upstream TaskTracker dataset construction "
            "uses BIPIA attack prompts."
        ),
    },

    "chatbot_instruction_prompts": {
        "semantic_type": "general_benign_instruction",
        "decision": "candidate_keep",
        "contamination_risk": "low",
        "reason": "General instruction-following corpus.",
    },

    "open-instruct": {
        "semantic_type": "general_benign_instruction",
        "decision": "candidate_keep",
        "contamination_risk": "low",
        "reason": "General instruction-following corpus.",
    },

    "Alpaca": {
        "semantic_type": "general_benign_instruction",
        "decision": "candidate_keep",
        "contamination_risk": "low",
        "reason": "General instruction-following corpus.",
    },

    "grok-conversation-harmless": {
        "semantic_type": "benign_conversation",
        "decision": "candidate_keep",
        "contamination_risk": "low",
        "reason": "Harmless conversational data.",
    },

    "ultrachat_200k": {
        "semantic_type": "general_benign_conversation",
        "decision": "candidate_keep",
        "contamination_risk": "low",
        "reason": "General conversational corpus.",
    },

    "no_robots": {
        "semantic_type": "general_benign_instruction",
        "decision": "candidate_keep",
        "contamination_risk": "low",
        "reason": "General instruction-following corpus.",
    },

    "safe-guard-prompt-injection": {
        "semantic_type": "prompt_injection",
        "decision": "candidate_keep",
        "contamination_risk": "low",
        "reason": "Prompt-injection classification dataset.",
    },

    "prompt-injections": {
        "semantic_type": "prompt_injection",
        "decision": "candidate_keep",
        "contamination_risk": "low",
        "reason": "Prompt-injection dataset.",
    },

    "Prompt-Injection-Mixed-Techniques": {
        "semantic_type": "prompt_injection",
        "decision": "candidate_keep",
        "contamination_risk": "medium",
        "reason": "Prompt-injection data with mixed techniques.",
    },

    "StruQ": {
        "semantic_type": "indirect_prompt_injection",
        "decision": "candidate_keep",
        "contamination_risk": "medium",
        "reason": (
            "Structured-query defense dataset with "
            "instructions inserted into untrusted data."
        ),
    },

    "jailbreak-classification": {
        "semantic_type": "jailbreak",
        "decision": "review",
        "contamination_risk": "low",
        "reason": (
            "Jailbreak is related but outside the primary "
            "indirect-prompt-injection threat model."
        ),
    },

    "vigil-jailbreak-ada-002": {
        "semantic_type": "jailbreak",
        "decision": "review",
        "contamination_risk": "low",
        "reason": "Jailbreak data; scope mismatch must be reviewed.",
    },

    "ChatGPT-Jailbreak-Prompts": {
        "semantic_type": "jailbreak",
        "decision": "review",
        "contamination_risk": "low",
        "reason": "Jailbreak data; scope mismatch must be reviewed.",
    },

    "hackaprompt-dataset": {
        "semantic_type": "prompt_hacking",
        "decision": "review",
        "contamination_risk": "medium",
        "reason": "Prompt-hacking data may mix several attack modes.",
    },

    "over-defense": {
        "semantic_type": "unknown_hard_benign",
        "decision": "review",
        "contamination_risk": "medium",
        "reason": (
            "Origin must be verified before use because "
            "NotInject is reserved for evaluation."
        ),
    },

    "LLM Augmented set": {
        "semantic_type": "synthetic_unknown",
        "decision": "review",
        "contamination_risk": "medium",
        "reason": "Synthetic provenance and generation process need review.",
    },

    "InjecAgent": {
        "semantic_type": "agent_prompt_injection",
        "decision": "review",
        "contamination_risk": "medium",
        "reason": "Potentially relevant agent injection data; inspect provenance.",
    },
}

def load_json(path: Path) -> list[dict]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON list."
        )
    return data

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

def audit_sources(
    records: list[dict],
    examples_per_label: int,
):
    source_stats = defaultdict(
        lambda: {
            "total": 0,
            "labels": Counter(),
        }
    )

    examples = defaultdict(
        lambda: {
            0: [],
            1: [],
        }
    )

    for index, record in enumerate(records):
        source = str(
            record.get(
                "source",
                "<missing>",
            )
        )

        label = record.get(
            "label"
        )

        prompt = record.get(
            "prompt"
        )

        source_stats[
            source
        ][
            "total"
        ] += 1

        source_stats[
            source
        ][
            "labels"
        ][
            str(label)
        ] += 1

        if (
            label in {0, 1}
            and isinstance(prompt, str)
            and prompt.strip()
            and len(
                examples[source][label]
            ) < examples_per_label
        ):

            examples[
                source
            ][
                label
            ].append(
                {
                    "index": index,
                    "source": source,
                    "label": label,
                    "prompt": prompt,
                }
            )

    report = []

    for source in sorted(
        source_stats
    ):

        stats = source_stats[
            source
        ]

        total = stats[
            "total"
        ]

        benign = stats[
            "labels"
        ].get(
            "0",
            0,
        )

        injection = stats[
            "labels"
        ].get(
            "1",
            0,
        )

        policy = SOURCE_POLICY.get(
            source,
            {
                "semantic_type": "unknown",
                "decision": "review",
                "contamination_risk": "unknown",
                "reason": "Provenance not yet verified.",
            },
        )

        report.append(
            {
                "source": source,
                "total": total,
                "benign": benign,
                "injection": injection,
                "benign_ratio": (
                    benign / total
                    if total
                    else 0.0
                ),
                "injection_ratio": (
                    injection / total
                    if total
                    else 0.0
                ),
                "semantic_type":
                    policy[
                        "semantic_type"
                    ],
                "decision":
                    policy[
                        "decision"
                    ],
                "contamination_risk":
                    policy[
                        "contamination_risk"
                    ],
                "reason":
                    policy[
                        "reason"
                    ],
                "provenance_verified":
                    False,
            }
        )

    example_records = []

    for source in sorted(
        examples
    ):

        for label in (
            0,
            1,
        ):

            example_records.extend(
                examples[
                    source
                ][
                    label
                ]
            )

    return (
        report,
        example_records,
    )

def summarize(
    report: list[dict],
):

    decision_counts = Counter()

    decision_samples = Counter()

    for item in report:

        decision = item[
            "decision"
        ]

        decision_counts[
            decision
        ] += 1

        decision_samples[
            decision
        ] += item[
            "total"
        ]

    return {
        "number_of_sources":
            len(report),

        "source_decisions":
            dict(
                decision_counts
            ),

        "sample_decisions":
            dict(
                decision_samples
            ),
    }


def print_report(
    report: list[dict],
    summary: dict,
) -> None:

    print(
        "\n================================"
    )

    print(
        "InjecGuard Source Audit"
    )

    print(
        "================================"
    )

    for item in report:

        print(
            f"\n{item['source']}"
        )

        print(
            f"  total: {item['total']}"
        )

        print(
            f"  benign: {item['benign']}"
        )

        print(
            f"  injection: "
            f"{item['injection']}"
        )

        print(
            f"  semantic: "
            f"{item['semantic_type']}"
        )

        print(
            f"  risk: "
            f"{item['contamination_risk']}"
        )

        print(
            f"  decision: "
            f"{item['decision']}"
        )

    print(
        "\n================================"
    )

    print(
        "Summary"
    )

    print(
        "================================"
    )

    print(
        "Sources:",
        summary[
            "number_of_sources"
        ],
    )

    for decision, count in (
        summary[
            "source_decisions"
        ].items()
    ):

        samples = (
            summary[
                "sample_decisions"
            ].get(
                decision,
                0,
            )
        )

        print(
            f"{decision}: "
            f"{count} sources, "
            f"{samples} samples"
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/raw/injecguard/"
            "train.json"
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
        "--examples-per-label",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    records = load_json(
        input_path
    )

    (
        source_report,
        examples,
    ) = audit_sources(
        records,
        args.examples_per_label,
    )

    summary = summarize(
        source_report
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        /
        "source_audit.json"
    ).write_text(
        json.dumps(
            {
                "summary": summary,
                "sources": source_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_jsonl(
        examples,
        output_dir
        /
        "source_examples.jsonl",
    )

    (
        output_dir
        /
        "source_registry_draft.json"
    ).write_text(
        json.dumps(
            source_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_report(
        source_report,
        summary,
    )

    print(
        "\nReport:",
        output_dir
        /
        "source_audit.json",
    )

if __name__ == "__main__":
    main()