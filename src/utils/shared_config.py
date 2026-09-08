import os

SEARCH_API_KEY_ENV_VARS = {
    'serper': 'SERPER_API_KEY',
    'tavily': 'TAVILY_API_KEY',
}

def validate_search_options(search_type: str, num_searches: int) -> None:
    if not isinstance(search_type, str) or search_type not in SEARCH_API_KEY_ENV_VARS:
        raise ValueError('search.type must be in the list.')
    if type(num_searches) is not int or num_searches < 1:
        raise ValueError('searche.num_searches must be a positive integer.')
    if search_type == 'tavily' and num_searches > 20:
        raise ValueError('search.num_searches must be between 1 and 20 for Tavily')

def get_search_api_key(search_type: str) -> str:
    validate_search_options(search_type, 1)
    return get_api_key(SEARCH_API_KEY_ENV_VARS[search_type])

def get_api_key(env_var_name: str) -> str:
    api_key = os.getenv(env_var_name, '').strip()
    if not api_key or api_key.lower().startswith('your_'):
        raise ValueError(f'Missing {env_var_name}. Set it before using this provider.')
    return api_key
    