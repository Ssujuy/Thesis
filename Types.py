from enum import Enum

DEFAULT_PROJECTION_DIMENSION = 768

class ProjectionState(Enum):

    NO_PROJECTION = 0
    NOT_TRAINABLE = 1
    TRAINABLE = 2

class HiddenState(Enum):

    CLS     = 0,
    MEAN    = 1,
    BOTH    = 2