from enum import Enum


class Database(Enum):
    NAME = "db_timeseries"


class DatabaseCollection(Enum):
    DATASETS = "datasets"
    MODELS = "models"
