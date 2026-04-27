"""HmValidateEmpty — check for empty, whitespace-only, or TODO content."""

import re

from experience.example.code_auto_encoder.harness_model.function.harness_validator_op import HarnessValidatorOp


class HmValidateEmpty(HarnessValidatorOp):
    name = "validate_empty"
    description = "Check content is not empty, whitespace-only, or TODO placeholder."

    def validate(self, content: str) -> tuple:
        stripped = content.strip()
        if not stripped:
            return (False, "content is empty")
        if stripped.upper() == "TODO":
            return (False, "content is TODO placeholder")
        # Check if it's mostly whitespace or common filler
        if re.fullmatch(r"[\s\n\r]*", stripped):
            return (False, "content is whitespace-only")
        return (True, "")

    def schema(self) -> str:
        return "validate_empty() — checks content is not empty or TODO."
