"""HmValidateLength — check line-count bounds."""

from experience.example.code_auto_encoder.harness_model.function.harness_validator_op import HarnessValidatorOp


class HmValidateLength(HarnessValidatorOp):
    name = "validate_length"
    description = "Check content line count is within acceptable bounds."

    def __init__(self, min_lines: int = 1, max_lines: int = 500):
        self.min_lines = min_lines
        self.max_lines = max_lines

    def validate(self, content: str) -> tuple:
        lines = content.splitlines()
        n = len(lines)
        if n < self.min_lines:
            return (False, f"content has {n} lines, minimum is {self.min_lines}")
        if n > self.max_lines:
            return (False, f"content has {n} lines, maximum is {self.max_lines}")
        return (True, "")

    def schema(self) -> str:
        return (
            f"validate_length(min_lines={self.min_lines}, max_lines={self.max_lines})"
            " — checks line count bounds."
        )
