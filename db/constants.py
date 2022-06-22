from enum import Enum


class Database(Enum):
    NAME = "timeseries"


class DatabaseCollection(Enum):
    DATASETS = "datasets"
    MODELS = "models"
