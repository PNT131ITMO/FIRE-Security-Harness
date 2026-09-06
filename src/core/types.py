import dataclasses

@dataclasses.dataclass()
class GoogleSearchResult:
    query: str
    result: str

@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str

class VerificationError(RuntimeError):
    def __init__(self, message: str, searches: dict, usage: dict):
        super().__init(message)
        self.searches = searches
        self.usage = usage

