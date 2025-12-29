"""Custom exceptions for the framework."""
from typing import Optional, Any


class HallucinationDetectionError(Exception):
    """Base framework exception."""
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(HallucinationDetectionError):
    """Configuration error."""
    pass


class DatasetError(HallucinationDetectionError):
    """Dataset operation error."""
    pass


class DatasetNotFoundError(DatasetError):
    """Dataset not found."""
    pass


class DatasetParseError(DatasetError):
    """Dataset parsing error."""
    pass


class ModelError(HallucinationDetectionError):
    """Model operation error."""
    pass


class ModelLoadError(ModelError):
    """Model loading error."""
    pass


class ModelInferenceError(ModelError):
    """Model inference error."""
    pass


class FeatureExtractionError(HallucinationDetectionError):
    """Feature extraction error."""
    pass


class MethodError(HallucinationDetectionError):
    """Detection method error."""
    pass


class MethodNotFittedError(MethodError):
    """Method not fitted."""
    pass


class EvaluationError(HallucinationDetectionError):
    """Evaluation error."""
    pass


class CacheError(HallucinationDetectionError):
    """Cache operation error."""
    pass


class ValidationError(HallucinationDetectionError):
    """Validation error."""
    pass


class PipelineError(HallucinationDetectionError):
    """Pipeline execution error."""
    pass


class PromptError(HallucinationDetectionError):
    """Prompt construction error."""
    pass
