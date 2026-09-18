from __future__ import annotations

from typing import TYPE_CHECKING
import logging

from src.core.types import GoogleSearchResult, FinalAnswer
from src.core import evidence_manager, query_generator, retriever
from src.utils import text_utils
from src.utils.similarity import get_sentence_similarity
from src.utils.shared_config import validate_search_options

if TYPE_CHECKING:
    from src.core.agent import Model

LOGGER = logging.getLogger(__name__)

def final_answer_or_next_search(
    atomic_claim: str,
    past_searches: list[GoogleSearchResult],
    model: Model,
    diverse_prompt: bool = False,
    tolerance: int = 4,
    search_type: str = 'serper',
    num_searches: int = 3,
) -> tuple[FinalAnswer | GoogleSearchResult | None | str, dict | None]:
    if type(tolerance) is not int or tolerance < 2:
        raise ValueError('tolerance must be an integer >= 2.')
    
    validate_search_options(search_type, num_searches)

    knowledge = evidence_manager.build_knowledge(past_searches)

    full_prompt = query_generator.build_query_or_answer_prompt(
        atomic_claim=atomic_claim,
        knowledge=knowledge
    )

    query_history = evidence_manager.get_query_history(past_searches)
    search_history = evidence_manager.get_search_history(past_searches)

    if diverse_prompt:
        if len(query_history) >= 2:
            full_prompt += "Please pay attention to optimizing the query to make it more diverse and the retrieved knowledge is as different as possible."

        if len(search_history) >= tolerance - 1 and get_sentence_similarity(search_history[-1],
                                                                            search_history[-(tolerance - 1):-1],
                                                                            threshold=0.9) >= tolerance - 2:
            full_prompt += "\n\nPlease note! We have detected multiple very similar contents in the Knowledge section. Please optimize your query so that the retrieved knowledge is as different as possible."

        if len(query_history) >= tolerance - 1 and get_sentence_similarity(query_history[-1],
                                                                           query_history[-(tolerance - 1):-1],
                                                                           threshold=0.9) >= tolerance - 2:
            full_prompt += "\nPlease note that we have detected very similar content many times in the past query history. Please pay attention to optimizing the query to make it more diverse."

    model_response, usage = model.generate(full_prompt)

    answer_or_next_query = text_utils.extract_json_from_output(model_response)

    if answer_or_next_query is None:
        LOGGER.info('Invalid model output: no JSON decision found.')
        return None, usage
    
    actions = {'final_answer', 'search_query'} & answer_or_next_query.keys()

    if len(actions) != 1:
        LOGGER.info('Invalid model output: expected exactly one answer or search query.')
        return None, usage
    
    if 'final_answer' in answer_or_next_query:
        answer = text_utils.normalize_label(answer_or_next_query['final_answer'])

        if answer is None:
            LOGGER.info('Invalid model output: final_answer must be True or False.')
            return None, usage
        
        LOGGER.info('Decision: final_answer = %s', answer)
        
        return FinalAnswer(
            response=model_response,
            answer=answer,
        ), usage
    
    if 'search_query' in answer_or_next_query:
        query = answer_or_next_query['search_query']

        if not isinstance(query, str) or not query.strip():
            LOGGER.info('Invalid model output: search_query must be a non-empty string.')
            return None, usage
        
        query = query.strip()

        LOGGER.info('Proposed search query: %s', query)

        if query_history:
            LOGGER.info('Checking query/evidence repetition with the embedding model...')
        
        if(
            len(query_history) >= tolerance - 1
            and get_sentence_similarity(
                query,
                query_history[-(tolerance - 1):],
                threshold=0.9,
            ) >= tolerance - 1
        ):
            LOGGER.info('Early stop: the proposed query repeats recent queries.')
            return '_Early_Stop', usage

        if (
            len(search_history) >= tolerance
            and get_sentence_similarity(
                search_history[-1],
                search_history[-tolerance:-1],
                threshold=0.9,
            ) >= tolerance - 1
        ):
            LOGGER.info('Early stop: recent searches return repeated evidence.')
            return '_Early_Stop', usage

        LOGGER.info('Searching %s (up to %s results)...', search_type.capitalize(), num_searches)

        search_result = retriever.call_search(
            search_query=query,
            search_type=search_type,
            num_searches=num_searches
        )

        LOGGER.info('Retrieved evidence:\n%s', search_result)

        return GoogleSearchResult(
            query=query,
            result=search_result,
        ), usage
    
    return None, usage