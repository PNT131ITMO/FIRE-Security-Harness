from pathlib import Path
import re

from src.utils import text_utils

_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'

PROMPT_DIR = Path(__file__).resolve().parents[1] / 'prompts'

def _read_prompt(file_name: str) -> str:
    return (PROMPT_DIR / file_name).read_text(encoding='utf-8')

def build_query_or_answer_prompt(atomic_claim: str, knowledge: str) -> str:
    return _build_prompt('query_or_answer.txt', atomic_claim, knowledge)

def build_final_answer_prompt(atomic_claim: str, knowledge: str) -> str:
    return _build_prompt('final_answer.txt', atomic_claim, knowledge)

def _build_prompt(file_name: str, atomic_claim: str, knowledge: str) -> str:
    values = {
        _STATEMENT_PLACEHOLDER: atomic_claim,
        _KNOWLEDGE_PLACEHOLDER: atomic_claim,
    }
    prompt = re.sub(
        r'\[STATEMENT\]\[KNOWLEDGE\]',
        lambda match: values[match.group()],
        _read_prompt(file_name),
    )
    return text_utils.strip_string(prompt)


