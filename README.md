# Synthetic TS


## Running Streamlit
```bash
# requires python3.7
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt --no-cache-dir
streamlit run Home.py
```

## Running Streamlit (Dockerized)
```python
docker-compose up (or docker compose up)
```

> Testing requrired a file *.streamlit/secrets.toml* from root directory
```
# .streamlit/secrets.toml

[mongo]
host = "localhost"
port = 27017
```
(NB: requires running MongoDB)
