import time
from typing import Any, Optional, Literal

import requests

from src.utils import shared_config
from src.utils.telemetry import increment
from src.core.documents import EvidenceDocument

_SERPER_URL = 'https://google.serper.dev'
_TAVILY_URL = 'https://api.tavily.com/search'
NO_RESULT_MSG = 'No good web search result was found'

def _post_search_json(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
        max_retries: int, 
        provider: str
) -> dict[str, Any]:
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError('max_retries must be a non-negative integer.')
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            if isinstance(error, requests.HTTPError):
                status = error.response.status_code if error.response is not None else None
                if status not in {408, 429} and (status is None or not 500 <= status < 600):
                    raise
            elif not isinstance(error, (requests.ConnectionError, requests.Timeout)):
                raise
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** attempt, 30))
        else:
            results = response.json()
            if not isinstance(results, dict):
                raise ValueError(f'{provider} returned a non-object JSON response')
            return results

class SerperAPI:
    def __init__(
            self,
            serper_api_key: str,
            gl: str = 'us',
            hl: str = 'en',
            k: int = 1,
            tbs: Optional[str] = None,
            search_type: Literal['news', 'search', 'places', 'images'] = 'search',
            timeout: float = 30,
    ):
        if type(k) is not int or k < 1:
            raise ValueError('k must be a positive integer.')
        if not isinstance(timeout, (int,  float)) or isinstance(timeout, bool) or not 0 < timeout < float('inf'):
            raise ValueError('timeout must be finite and positive.')
        self.serper_api_key = serper_api_key
        self.gl = gl
        self.hl = hl
        self.k = k
        self.tbs = tbs
        self.search_type = search_type
        self.timeout = timeout

        self.result_key_for_type = {
            'news': 'news',
            'places': 'places',
            'images': 'images',
            'search': 'organic',
        }
        if search_type not in self.result_key_for_type:
            raise ValueError(f'Unsupported Serper search type: {search_type}')

    def run(self, query: str, **kwargs: Any) -> str:
        if (not isinstance(self.serper_api_key, str) or not self.serper_api_key.strip()
            or self.serper_api_key.strip().lower().startswith('your_')):
            raise ValueError('Missing SERPER_API_KEY. Set it before searching.')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a non-empty string.')

        results = self._google_serper_api_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            tbs=self.tbs,
            search_type=self.search_type,
            **kwargs,
        )

        return self._parse_results(results)

    def _google_serper_api_results(
        self,
        search_term: str,
        search_type: str = 'search',
        max_retries: int = 3,
        **kwargs: Any,
    ) -> dict[Any, Any]:
        headers = {
            'X-API-KEY': self.serper_api_key or '',
            'Content-Type': 'application/json',
        }

        params = {
            'q': search_term,
            **{key: value for key, value in kwargs.item() if value is not None},
        }

        return _post_search_json(
            f'{_SERPER_URL}/{search_type}', headers, params,
            self.timeout, max_retries, 'Serper',
        )

    def _parse_snippets(self, results: dict[Any, Any]) -> list[str]:
        snippets = []

        if results.get('answerBox'):
            answer_box = results.get('answerBox', {})
            answer = answer_box.get('answer')
            snippet = answer_box.get('snippet')
            snippet_highlighted = answer_box.get('snippetHighlighted')

            if answer and isinstance(answer, str):
                snippets.append(answer)

            if snippet and isinstance(snippet, str):
                snippets.append(snippet.replace('\n', ' '))

            if snippet_highlighted:
                if isinstance(snippet_highlighted, str):
                    snippets.append(snippet_highlighted)
                elif isinstance(snippet_highlighted, list):
                    snippets.extend(s for s in snippet_highlighted if isinstance(s, str) and s.strip())

        if results.get('knowledgeGraph'):
            kg = results.get('knowledgeGraph', {})
            title = kg.get('title')
            entity_type = kg.get('type')
            description = kg.get('description')

            if entity_type:
                snippets.append(f'{title}: {entity_type}.')

            if isinstance(description, str) and description.strip():
                snippets.append(description)

            for attribute, value in (kg.get('attribute') or {}).items():
                snippets.append(f'{title} {attribute}: {value}.')

        result_key = self.result_key_for_type[self.search_type]

        if result_key in results:
            for result in (results[result_key] or [])[:self.k]:
                if not isinstance(result, dict):
                    continue
                if isinstance(result.get('snippet'), str) and result['snippet'].strip():
                    snippets.append(result['snippet'])

                for attribute, value in (result.get('attributes') or {}).items():
                    snippets.append(f'{attribute}: {value}.')

        if not snippets:
            return [NO_RESULT_MSG]

        return snippets

    def _parse_results(self, results: dict[Any, Any]) -> str:
        return ' '.join(self._parse_snippets(results))

