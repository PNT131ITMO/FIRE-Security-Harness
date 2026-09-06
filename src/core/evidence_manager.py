import dataclasses

from src.core.types import GoogleSearchResult

def build_knowledge(past_searches: list[GoogleSearchResult]) -> str:
    knowledge = '\n'.join([s.result for s in past_searches])
    return 'N/A' if not knowledge else knowledge

def build_search_dicts(searches: list[GoogleSearchResult]) -> dict:
    return {
        'google_searches': [dataclasses.asdict(s) for s in searches]
    }

def get_query_history(past_searches: list[GoogleSearchResult]) -> list[str]:
    return [item.query for item in past_searches]

def get_search_history(past_searches: list[GoogleSearchResult]) -> list[str]:
    return [item.result for item in past_searches]