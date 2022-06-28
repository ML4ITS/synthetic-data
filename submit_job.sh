#!/usr/bin/env sh

RAY_HOST=$(grep RAY_HOST .env | cut -d "=" -f2)
RAY_PORT=$(grep RAY_PORT .env | cut -d "=" -f2)
export RAY_ADDRESS="http://$RAY_HOST:$RAY_PORT"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": ["pymongo", "python-dotenv"]}' -- python3 raytune.py
