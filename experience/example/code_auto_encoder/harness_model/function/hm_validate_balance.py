"""HmValidateBalance — check brace/paren/bracket balance."""

from experience.example.code_auto_encoder.harness_model.function.harness_validator_op import HarnessValidatorOp


class HmValidateBalance(HarnessValidatorOp):
    name = "validate_balance"
    description = "Check brace {}, paren (), bracket [] balance."

    PAIRS = {"{": "}", "(": ")", "[": "]"}

    def validate(self, content: str) -> tuple:
        stack = []
        for ch in content:
            if ch in self.PAIRS:
                stack.append(ch)
            elif ch in self.PAIRS.values():
                if not stack:
                    return (False, f"unmatched closing '{ch}'")
                opener = stack.pop()
                expected = self.PAIRS[opener]
                if ch != expected:
                    return (False, f"mismatched brace: '{opener}' vs '{ch}'")
        if stack:
            return (False, f"unmatched opening '{stack[-1]}'")
        return (True, "")

    def schema(self) -> str:
        return "validate_balance() — checks {} () [] are balanced."
