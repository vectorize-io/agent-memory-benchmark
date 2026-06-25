# Deterministic grading environment for sdebench tasks.
# The ratelimiter lib is dependency-free; pytest + git is all we need.
FROM python:3.11-slim
RUN pip install --no-cache-dir pytest==8.* && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
