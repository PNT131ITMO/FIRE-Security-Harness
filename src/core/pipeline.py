from __future__ import annotations

from typing import Any, TYPE_CHECKING
import logging

from src.core.types import FinalAnswer, GoogleSearchResult, VerificationError
from src.core import decision_module, evidence_manager, verifier
from src.utils.shared_config import validate_search_options

if TYPE_CHECKING:
    from src.core.agent import Model

LOGGER = logging.getLogger(__name__)

class _UsageTrackingModel:
    def __init__(self, model: Model):
        self.model = model
        self.usage = {'input_tokens': 0, 'output_tokens': 0, 'unreported_calls': 0}

    def generate(self, context:str) -> tuple[str, dict | None]:
        try:
            response, usage = self.model.generate(context)
        except Exception:
            self.usage['unreported_calls'] += 1
            raise
        
        usage = usage or {}

        for key in ('input_tokens', 'output_tokens'):
            self.usage[key] += usage.get(key) or 0
        if any(usage.get(key) is None for key in ('input_tokens', 'output_tokens')):
            self.usage['unreported_calls'] += 1
        
        return response, usage

def verify_atomic_claim(
    atomic_claim: str,
    rater: Model,
    max_steps: int = 5,
    max_retries: int = 10,
    diverse_prompt: bool = False,
    tolerance: int = 2,
    search_type: str = 'serper',
    num_searches: int = 3,
) -> tuple[FinalAnswer | None, dict[str, Any], dict[str, int]]:
    if not isinstance(atomic_claim, str) or not atomic_claim.split():
        raise ValueError('atomic_claim must be a non-empty string.')
    
    for name, value, minimum in (
        ('max_steps', max_steps, 0),
        ('max_retries', max_retries, 0),
        ('tolerance', tolerance, 2),
        ('num_searches',num_searches, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    
    if not isinstance(diverse_prompt, bool):
        raise ValueError('diverse_prompt must be a boolean.')
    
    validate_search_options(search_type, num_searches)

    searches: list[GoogleSearchResult] = []
    tracked_model = _UsageTrackingModel(rater)
    LOGGER.info('Claim: %s', atomic_claim)

    fallback_reason = 'maximum decision rounds reached' if max_steps else 'no retrieval rounds configured'

    try:
        for step in range(max_steps):
            decision = None

            for attempt in range(max_retries + 1):
                LOGGER.info('\nStep %s/%s | decision attempt %s/%s, completed searches: %s',
                step + 1, max_steps, attempt + 1, max_retries + 1, len(searches))

                decision, _ = decision_module.final_answer_or_next_search(
                    atomic_claim=atomic_claim,
                    past_searches=searches,
                    diverse_prompt=diverse_prompt,
                    model=tracked_model,
                    tolerance=tolerance,
                    search_type=search_type,
                    num_searches=num_searches,
                )
                if decision is not None:
                    break
            
            if isinstance(decision, FinalAnswer):
                return decision, evidence_manager.build_search_dicts(searches), tracked_model.usage
            
            if isinstance(decision, GoogleSearchResult):
                searches.append(decision)
            else:
                fallback_reason = ('repeated query or evidence' if decision == '_Early_Stop'
                                    else 'decision output retries exhausted.')
                break
        
        LOGGER.info('\nFinal verification: %s.', fallback_reason)

        for attempt in range(max_retries + 1):
            LOGGER.info('Final answer attempt %s/%s...', attempt + 1, max_retries + 1)
            final_answer, _ = verifier.must_get_final_answer(
                atomic_fact=atomic_claim, searches=searches, model=tracked_model,
            )

            if final_answer is not None:
                LOGGER.info('Decision: final_answer = %s', final_answer.answer)
                return final_answer, evidence_manager.build_search_dicts(searches), tracked_model.usage
            
            LOGGER.info('Invalid final answer; no binary verdict was returned.')
    except Exception as error:
        raise VerificationError(
            f'{type(error).__name__}: {error}',
            searches=evidence_manager.build_search_dicts(searches),
            usage=tracked_model.usage.copy(),
        ) from error
    
    return None, evidence_manager.build_search_dicts(searches), tracked_model.usage
