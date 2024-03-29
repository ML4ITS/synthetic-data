FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN echo "export LC_ALL=$LC_ALL" >> /etc/profile.d/locale.sh
RUN echo "export LANG=$LANG" >> /etc/profile.d/locale.sh

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev

RUN pip install -U pip
COPY requirements.txt ./
COPY setup.py ./
RUN pip install -e .
ENV PYTHONUNBUFFERED 1
COPY synthetic_data /synthetic_data
COPY .env /synthetic_data/.env

WORKDIR /synthetic_data
