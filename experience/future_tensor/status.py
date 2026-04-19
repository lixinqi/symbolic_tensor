"""
Status :=
    Oneof
    | $confidence Bounded0To1[float]
    | $self_confidence_yet_failed Bounded0To1Exclude0[float]
    | $kConfidenceNotBounded 2
    | $kContextOverflow 3

Status.convert_status_to_float :=
    float
    <- $status Status
    # inline
    <- { match status }
    <- { case $confidence: return confidence }
    <- { case $self_confidence_yet_failed: return -self_confidence_yet_failed }
    <- { otherwise $errno: return -enum_value(errno)}

Status.convert_float_to_status :=
    $status Status
    <- $value float
    # inline
    <- { reverse procedure of convert_status_to_float }
"""

from typing import Union


class Status:
    """Tagged union for FutureTensor element confidence/status.

    Variants:
        confidence(float)              — success, value in [0.0, 1.0]
        self_confidence_yet_failed(float) — failed with self-assessed confidence, value in (0.0, 1.0]
        kConfidenceNotBounded          — error: confidence not in range (enum value 2)
        kContextOverflow               — error: context too long (enum value 3)
    """

    # Enum sentinel values
    kConfidenceNotBounded = None  # set below as class instance
    kContextOverflow = None       # set below as class instance

    _ENUM_CONFIDENCE_NOT_BOUNDED = 2
    _ENUM_CONTEXT_OVERFLOW = 3

    def __init__(self, tag: str, value: float = 0.0):
        self._tag = tag
        self._value = value

    @staticmethod
    def confidence(value: float) -> "Status":
        """Create a confidence status (success). Value clamped to [0.0, 1.0]."""
        v = max(0.0, min(1.0, value))
        return Status("confidence", v)

    @staticmethod
    def self_confidence_yet_failed(value: float) -> "Status":
        """Create a self_confidence_yet_failed status. Value clamped to (0.0, 1.0]."""
        v = max(1e-10, min(1.0, value))
        return Status("self_confidence_yet_failed", v)

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def value(self) -> float:
        return self._value

    @property
    def is_confidence(self) -> bool:
        return self._tag == "confidence"

    @property
    def is_self_confidence_yet_failed(self) -> bool:
        return self._tag == "self_confidence_yet_failed"

    @property
    def is_kConfidenceNotBounded(self) -> bool:
        return self._tag == "kConfidenceNotBounded"

    @property
    def is_kContextOverflow(self) -> bool:
        return self._tag == "kContextOverflow"

    @staticmethod
    def convert_status_to_float(status: "Status") -> float:
        """Convert Status to float for tensor storage.

        confidence             →  value       (0.0 to 1.0)
        self_confidence_yet_failed → -value   (-1.0 to ~0.0)
        kConfidenceNotBounded  → -2.0
        kContextOverflow       → -3.0
        """
        if status._tag == "confidence":
            return status._value
        if status._tag == "self_confidence_yet_failed":
            return -status._value
        if status._tag == "kConfidenceNotBounded":
            return -float(Status._ENUM_CONFIDENCE_NOT_BOUNDED)
        if status._tag == "kContextOverflow":
            return -float(Status._ENUM_CONTEXT_OVERFLOW)
        raise ValueError(f"Unknown status tag: {status._tag}")

    @staticmethod
    def convert_float_to_status(value: float) -> "Status":
        """Reverse of convert_status_to_float.

        [0.0, 1.0]    → confidence
        (-1.0, ~0.0)  → self_confidence_yet_failed (negate)
        -2.0           → kConfidenceNotBounded
        -3.0           → kContextOverflow
        """
        if value >= 0.0:
            return Status.confidence(value)
        if value == -float(Status._ENUM_CONFIDENCE_NOT_BOUNDED):
            return Status("kConfidenceNotBounded", float(Status._ENUM_CONFIDENCE_NOT_BOUNDED))
        if value == -float(Status._ENUM_CONTEXT_OVERFLOW):
            return Status("kContextOverflow", float(Status._ENUM_CONTEXT_OVERFLOW))
        # Negative but not an enum sentinel → self_confidence_yet_failed
        return Status.self_confidence_yet_failed(-value)

    def __repr__(self) -> str:
        if self._tag in ("confidence", "self_confidence_yet_failed"):
            return f"Status.{self._tag}({self._value})"
        return f"Status.{self._tag}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Status):
            return False
        return self._tag == other._tag and abs(self._value - other._value) < 1e-7


# Singleton enum variants
Status.kConfidenceNotBounded = Status("kConfidenceNotBounded", float(Status._ENUM_CONFIDENCE_NOT_BOUNDED))
Status.kContextOverflow = Status("kContextOverflow", float(Status._ENUM_CONTEXT_OVERFLOW))


if __name__ == "__main__":
    print("Running tests for Status...\n")

    def run_test(name, cond, expected=None, actual=None):
        if cond:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None:
                print(f"    expected: {expected}, actual: {actual}")

    # confidence
    s = Status.confidence(0.8)
    run_test("confidence tag", s.tag == "confidence")
    run_test("confidence value", abs(s.value - 0.8) < 0.01)
    run_test("confidence is_confidence", s.is_confidence)
    run_test("confidence to_float", abs(Status.convert_status_to_float(s) - 0.8) < 0.01)

    # self_confidence_yet_failed
    s = Status.self_confidence_yet_failed(0.6)
    run_test("scyf tag", s.tag == "self_confidence_yet_failed")
    run_test("scyf value", abs(s.value - 0.6) < 0.01)
    run_test("scyf to_float", abs(Status.convert_status_to_float(s) - (-0.6)) < 0.01)

    # kConfidenceNotBounded
    run_test("kCNB to_float", Status.convert_status_to_float(Status.kConfidenceNotBounded) == -2.0)

    # kContextOverflow
    run_test("kCO to_float", Status.convert_status_to_float(Status.kContextOverflow) == -3.0)

    # Round-trip
    for val in [0.0, 0.5, 1.0]:
        s = Status.confidence(val)
        f = Status.convert_status_to_float(s)
        s2 = Status.convert_float_to_status(f)
        run_test(f"roundtrip confidence({val})", s == s2)

    for val in [0.1, 0.5, 1.0]:
        s = Status.self_confidence_yet_failed(val)
        f = Status.convert_status_to_float(s)
        s2 = Status.convert_float_to_status(f)
        run_test(f"roundtrip scyf({val})", s == s2)

    s2 = Status.convert_float_to_status(-2.0)
    run_test("roundtrip kCNB", s2.is_kConfidenceNotBounded)

    s2 = Status.convert_float_to_status(-3.0)
    run_test("roundtrip kCO", s2.is_kContextOverflow)

    # Clamping
    s = Status.confidence(1.5)
    run_test("confidence clamp >1", abs(s.value - 1.0) < 0.01)
    s = Status.confidence(-0.5)
    run_test("confidence clamp <0", abs(s.value) < 0.01)

    print("\nAll tests completed.")
