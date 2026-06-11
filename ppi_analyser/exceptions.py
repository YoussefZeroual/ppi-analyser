class PPIAnalyserError(Exception):
    """Base exception for all ppi_analyser errors."""

class ProviderNotFoundError(PPIAnalyserError):
    """Raised when an unknown model provider is requested."""

class OllamaConnectionError(PPIAnalyserError):
    """Raised when the Ollama server cannot be reached."""

class ModelRateLimitError(PPIAnalyserError):
    """Raised when a provider's API quota is exceeded."""

class PPITagMissingError(PPIAnalyserError):
    """Raised when a sentence is missing <PPI></PPI> tags."""

class PromptParsingError(PPIAnalyserError):
    """Raised when a prompt_type cannot be extracted from a system prompt."""

class SentenceIndexError(PPIAnalyserError):
    """Raised when a requested sentence index exceeds the file size."""

class ConfigurationError(PPIAnalyserError):
    """Raised when required environment variables or config values are missing."""

class SegmentDetectionError(PPIAnalyserError):
    """Raised when dialogue/narration segment detection fails."""

class OutputExportError(PPIAnalyserError):
    """Raised when writing Excel, PDF, or HTML output fails."""
