import time
from typing import Any, Optional, Literal

import requests

from src.utils import shared_config
from src.utils.telemetry import increment
from src.core.documents import EvidenceDocument

_SERPER_URL = 'https://google.serper.dev'
_TAVILY_URL = 'https://api.tavily.com/search'
NO_RESULT_MSG = 'No good web search result was found'
