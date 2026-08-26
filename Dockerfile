# CPU-only image for the inm core pipeline (preprocess -> model -> postprocess
# -> analyze).
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .

WORKDIR /app/examples/cullin_example
ENTRYPOINT ["snakemake", "--cores", "all"]
CMD ["all"]
