# llm package

class QuotaExhaustedError(Exception):
    """Raised when all API keys for a provider have exhausted their quota or failed."""
    pass
