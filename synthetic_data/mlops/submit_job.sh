#!/usr/bin/env sh
MONGO_HOST=$(grep MONGO_HOST .env | cut -d "=" -f2)
MONGO_PORT=$(grep MONGO_PORT .env | cut -d "=" -f2)
MONGO_USERNAME=$(grep MONGO_USERNAME .env | cut -d "=" -f2)
MONGO_PASSWORD=$(grep MONGO_PASSWORD .env | cut -d "=" -f2)
RAY_HOST=$(grep RAY_HOST .env | cut -d "=" -f2)
RAY_PORT=$(grep RAY_PORT .env | cut -d "=" -f2)
ML_HOST=$(grep ML_HOST .env | cut -d "=" -f2)
ML_PORT=$(grep ML_PORT .env | cut -d "=" -f2)
FLASK_HOST=$(grep FLASK_HOST .env | cut -d "=" -f2)
FLASK_PORT=$(grep FLASK_PORT .env | cut -d "=" -f2)

export RAY_ADDRESS="http://$RAY_HOST:$RAY_PORT"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": ["pymongo", "python-dotenv"], "env_vars": {"MONGO_HOST": "'$MONGO_HOST'", "MONGO_PORT": "'$MONGO_PORT'", "MONGO_USERNAME": "'$MONGO_USERNAME'", "MONGO_PASSWORD": "'$MONGO_PASSWORD'", "ML_HOST": "'$ML_HOST'", "ML_PORT": "'$ML_PORT'", "FLASK_HOST": "'$FLASK_HOST'", "FLASK_PORT": "'$FLASK_PORT'"}}' -- python3 mlops/raytune.py