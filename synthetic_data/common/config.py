import os
from urllib.parse import quote_plus

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class ServerConfig:
    @property
    def APPLICATION_HOST(self):
        return os.getenv("APPLICATION_HOST")

    @property
    def COMPUTATION_HOST(self):
        return os.getenv("COMPUTATION_HOST")


class DatabaseConfig:
    @property
    def DATABASE_NAME(self):
        return os.getenv("DATABASE_NAME")

    @property
    def DATABASE_USERNAME(self):
        return quote_plus(os.getenv("DATABASE_USERNAME"))

    @property
    def DATABASE_PASSWORD(self):
        return quote_plus(os.getenv("DATABASE_PASSWORD"))

    @property
    def DATABASE_HOST(self):
        return os.getenv("DATABASE_HOST")

    @property
    def DATABASE_PORT(self):
        return int(os.getenv("DATABASE_PORT"))

    @property
    def URI_DATABASE(self):
        return f"mongodb://{self.DATABASE_USERNAME}:{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}?authSource=admin"


class ModelRegistryConfig(ServerConfig):
    @property
    def MODELREG_PORT(self):
        return int(os.getenv("MODELREG_PORT"))

    @property
    def URI_MODELREG_REMOTE(self):
        return f"http://{self.APPLICATION_HOST}:{self.MODELREG_PORT}"


class BackendConfig(ServerConfig):
    @property
    def BACKEND_HOST(self):
        return os.getenv("BACKEND_HOST")

    @property
    def BACKEND_PORT(self):
        return int(os.getenv("BACKEND_PORT"))

    @property
    def URI_BACKEND_LOCAL(self):
        return f"http://{self.BACKEND_HOST}:{self.BACKEND_PORT}"

    @property
    def URI_BACKEND_REMOTE(self):
        return f"http://{self.APPLICATION_HOST}:{self.BACKEND_PORT}"


class LocalConfig(BackendConfig, DatabaseConfig, ModelRegistryConfig):
    pass


class RemoteConfig(BackendConfig, ModelRegistryConfig):
    pass


if __name__ == "__main__":

    local = LocalConfig()
    remote = RemoteConfig()
