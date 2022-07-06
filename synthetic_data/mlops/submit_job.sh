#!/usr/bin/env sh
APPLICATION_SERVER=$(grep XXXXXXXXXXX .env | cut -d "=" -f2)
COMPUTATION_SERVER=$(grep XXXXXXXXXXX .env | cut -d "=" -f2)

RAY_PORT=$(grep RAY_PORT .env | cut -d "=" -f2)
MODELREG_PORT=$(grep MODELREG_PORT .env | cut -d "=" -f2)
BACKEND_PORT=$(grep BACKEND_PORT .env | cut -d "=" -f2)

export RAY_ADDRESS="http://$COMPUTATION_SERVER:$RAY_PORT"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": ["pymongo", "python-dotenv"], "env_vars": {"APPLICATION_SERVER": "'$APPLICATION_SERVER'", "MODELREG_PORT": "'$MODELREG_PORT'", "BACKEND_PORT": "'$BACKEND_PORT'"}}' -- python3 -m synthetic_data.mlops.raytune