from .harness_op import HarnessOp
from .hm_glob import HmGlob
from .hm_grep import HmGrep
from .hm_read import HmRead
from .hm_write import HmWrite

from .harness_validator_op import HarnessValidatorOp
from .hm_validate_tool_result import HmValidateToolResult
from .hm_validate_syntax import HmValidateSyntax
from .hm_validate_empty import HmValidateEmpty
from .hm_validate_length import HmValidateLength
from .hm_validate_balance import HmValidateBalance
from .hm_validate_result_gather import HmValidateResultGather

ALL_OPS = {
    "glob": HmGlob(),
    "grep": HmGrep(),
    "read": HmRead(),
    "write": HmWrite(),
}

ALL_VALIDATORS = {
    "validate_tool_result": HmValidateToolResult(),
    "validate_syntax": HmValidateSyntax(),
    "validate_empty": HmValidateEmpty(),
    "validate_length": HmValidateLength(),
    "validate_balance": HmValidateBalance(),
    "validate_result_gather": HmValidateResultGather,
}
