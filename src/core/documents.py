from dataclasses import dataclass
import hashlib
from urllib.parse import urlsplit

@dataclass(frozen=True)
class EvidenceDocument:
    text: str
    url: str = ''
    title: str = ''
    provider: str = 'unknown'

    def __post_init__(self):
        if not all(isinstance(value, str) for value in (self.text, self.url, self.title, self.provider)):
            raise ValueError('Evidence text, URL, title and provider must be strings.')

    @property
    def source(self) -> str:
        try:
            parsed = urlsplit(self.url)
            host = parsed.hostname or ''
            if parsed.scheme not in {'http', 'https'} or not host.isascii():
                return ''
            if not all(c.isalnum() or c in '.-' for c in host):
                return ''
            return host.lower()[:253]
        except ValueError:
            return ''

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()[:16]


def documents_from_records(records: list[dict]) -> list[EvidenceDocument]:
    if not isinstance(records, list) or not records:
        raise ValueError('evidence must be a non-empty list of source objects.')
    documents = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('text'), str) or not record['text'].strip():
            raise ValueError('Each evidence source must have non-empty text.')
        documents.append(EvidenceDocument(**{key: record[key] for key in
                         ('text', 'url', 'title', 'provider') if key in record}))
    return documents
