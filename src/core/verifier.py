from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.types import GoogleSearchResult, FinalAnswer
from src.core import evidence_manager, query_generator
from src.utils import text_utils

if TYPE_CHECKING:
    from src.core.agent import Model

def must_get_final_answer(
    atomic_fact: str,
    searches: list[GoogleSearchResult],
    model: Model,
) -> tuple[FinalAnswer | None, dict | None]:
    full_prompt = query_generator.build_final_answer_prompt(
        atomic_claim=atomic_fact,
        knowledge=evidence_manager.build_knowledge(searches),
    )

    model_response, usage = model.generate(full_prompt)

    answer = text_utils.extract_json_from_output(model_response)

    if not answer or 'search_query' in answer:
        return None, usage
    
    final_answer = text_utils.normalize_label(answer.get('final_answer'))

    if final_answer is None:
        return None, usage
    
    return FinalAnswer(response=model_response, answer=final_answer), usage