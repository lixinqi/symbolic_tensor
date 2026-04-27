"""HmValidateToolResult — check HarnessOp output for emptiness/errors."""

from experience.example.code_auto_encoder.harness_model.function.harness_validator_op import HarnessValidatorOp


class HmValidateToolResult(HarnessValidatorOp):
    name = "validate_tool_result"
    description = "Check tool output for errors, emptiness, or no-match indicators."

    ERROR_PREFIXES = (
        "(file not found:",
        "(read error:",
        "(regex error:",
        "(no matches)",
        "(empty file)",
        "(wrote 0 chars",
    )

    def validate(self, content: str) -> tuple:
        stripped = content.strip()
        if not stripped:
            return (False, "tool output is empty")
        for prefix in self.ERROR_PREFIXES:
            if stripped.startswith(prefix):
                return (False, f"tool output indicates failure: {stripped}")
        return (True, "")

    def schema(self) -> str:
        return "validate_tool_result() — checks for empty/error tool output."
