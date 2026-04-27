"""HmValidateSyntax — validate Python code with ast.parse."""

import ast

from experience.example.code_auto_encoder.harness_model.function.harness_validator_op import HarnessValidatorOp


class HmValidateSyntax(HarnessValidatorOp):
    name = "validate_syntax"
    description = "Check Python syntax using ast.parse."

    def validate(self, content: str) -> tuple:
        try:
            ast.parse(content)
            return (True, "")
        except SyntaxError as e:
            return (False, f"SyntaxError: {e.msg} at line {e.lineno}")

    def schema(self) -> str:
        return "validate_syntax() — checks Python syntax with ast.parse."
