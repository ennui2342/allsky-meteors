# Meteor scorer for a SyncAPI receiver (the cluster). Runs one pass and
# exits: `run` (new frames since the last pass, the default), or any other
# scorer command. Configure with INDI_DB, INDI_DB_PASSWORD,
# ALLSKY_METEORS_CONFIG, ALLSKY_METEORS_HOME and TZ (README).
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY allsky_meteors/ allsky_meteors/
COPY data/ data/
RUN chmod -R a+rX /app

RUN useradd --uid 1000 --create-home meteors
USER 1000
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "allsky_meteors.scorer"]
CMD ["run"]
