import os
from urllib.parse import quote_plus

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class Config:

    FLASK_HOST = os.getenv("FLASK_HOST")
    FLASK_PORT = os.getenv("FLASK_PORT")

    MONGO_DATABASE = os.getenv("MONGO_DATABASE")
    MONGO_HOST = os.getenv("MONGO_HOST")
    MONGO_PORT = int(os.getenv("MONGO_PORT"))
    MONGO_USERNAME = quote_plus(os.getenv("MONGO_USERNAME"))
    MONGO_PASSWORD = quote_plus(os.getenv("MONGO_PASSWORD"))

    MLFLOW_HOST = os.getenv("ML_HOST")
    MLFLOW_PORT = int(os.getenv("ML_PORT"))

    @property
    def MONGO_URI(self):
        return f"mongodb://{self.MONGO_HOST}:{self.MONGO_PORT}/{self.MONGO_DATABASE}"

    @property
    def MONGO_URI_WITH_AUTH(self):
        return f"mongodb://{self.MONGO_USERNAME}:{self.MONGO_PASSWORD}@{self.MONGO_HOST}:{self.MONGO_PORT}/{self.MONGO_DATABASE}?authSource=admin"

    @property
    def MLFLOW_TRACKING_URI(self):
        return f"http://{self.MLFLOW_HOST}:{self.MLFLOW_PORT}"

    @property
    def MLFLOW_REGISTRY_URI(self):
        return f"http://{self.MLFLOW_HOST}:{self.MLFLOW_PORT}"

    @property
    def API_URI(self):
        return f"http://{self.FLASK_HOST}:{self.FLASK_PORT}"

    def print_debug(self):
        print("---------------------------")
        print("# CONFIGURATION SETTINGS --")
        print("- MONGO_NAME    : ", self.MONGO_DATABASE)
        print("- MONGO_HOST    : ", self.MONGO_HOST)
        print("- MONGO_PORT    : ", self.MONGO_PORT)
        print("- MONGO_USER    : ", self.MONGO_USERNAME)
        print("- MONGO_PASS    : ", self.MONGO_PASSWORD)
        print("- MLFLOW_HOST   : ", self.MLFLOW_HOST)
        print("- MLFLOW_PORT   : ", self.MLFLOW_PORT)
        print("- FLASK_HOST    : ", self.FLASK_HOST)
        print("- FLASK_PORT    : ", self.FLASK_PORT)
        print("---------------------------")
