from enum import IntEnum, unique

@unique
class ErrCode(IntEnum):
    SUCCESS = 0
    FMT_ERR = -1
    WRONG_ANSWER = -2
    TIMEOUT = -3
    RUNTIME_ERR = - 4
    COMPILE_ERR= -5
    GROUND_TRUTH_ERR = -6
    