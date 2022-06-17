from enum import Enum


class ProcessType(Enum):
    HARMONIC = "Harmonic"
    GAUSSIAN_PROCESS = "GaussianProcess"
    PSEUDO_PERIODIC = "PseudoPeriodic"
    AUTO_REGRESSIVE = "AutoRegressive"
    CAR = "CAR"
    NARMA = "NARMA"


class ProcessKernel(Enum):
    CONSTANT = "Constant"
    EXPONENTIAL = "Exponential"
    SE = "SE"
    RQ = "RQ"
    LINEAR = "Linear"
    MATERN = "Matern"
    PERIODIC = "Periodic"
