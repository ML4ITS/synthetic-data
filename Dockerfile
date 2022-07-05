FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

RUN echo "export LC_ALL=$LC_ALL" >> /etc/profile.d/locale.sh
RUN echo "export LANG=$LANG" >> /etc/profile.d/locale.sh

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev

RUN pip install -U pip
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY synthetic_data /synthetic_data
COPY .env /synthetic_data/.env

WORKDIR /synthetic_data
