"""HarnessValidatorOp base class for deterministic validation operators.

A validator checks text content and returns (is_valid, message).
Pass → (True, ""). Fail → (False, error_description).
"""


class HarnessValidatorOp:
    """Base class for deterministic content validators."""

    name: str = ""
    description: str = ""

    def validate(self, content: str) -> tuple:
        """Validate content. Returns (is_valid: bool, message: str)."""
        raise NotImplementedError

    def schema(self) -> str:
        """Validator schema string for documentation."""
        raise NotImplementedError
