# Synthetic TS


## Running Streamlit
```bash
# requires python3.7
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt --no-cache-dir
streamlit run index.py
```

## Running Streamlit (Dockerized)
```bash
docker-compose up
```

### Format

Run `docker-compose run streamlit black src/`