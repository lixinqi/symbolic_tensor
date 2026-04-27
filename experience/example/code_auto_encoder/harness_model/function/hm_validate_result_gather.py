"""HmValidateResultGather — composite validator taking leafs as __init__ args."""

from typing import List

from experience.example.code_auto_encoder.harness_model.function.harness_validator_op import HarnessValidatorOp


class HmValidateResultGather(HarnessValidatorOp):
    name = "validate_result_gather"
    description = "Composite validator: runs all sub-validators, collects failures."

    def __init__(self, *validators: HarnessValidatorOp):
        self.validators: List[HarnessValidatorOp] = list(validators)

    def validate(self, content: str) -> tuple:
        failures = []
        for v in self.validators:
            ok, msg = v.validate(content)
            if not ok:
                failures.append(f"[{v.name}] {msg}")
        if failures:
            return (False, "\n".join(failures))
        return (True, "")

    def schema(self) -> str:
        names = ", ".join(v.name for v in self.validators)
        return f"validate_result_gather({names}) — composite validator."
