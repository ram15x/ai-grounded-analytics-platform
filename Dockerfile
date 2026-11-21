FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY semantic ./semantic
COPY config ./config
COPY demo ./demo
COPY corpus ./corpus
RUN pip install --no-cache-dir -e .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Recorded-runs view by default (no key, no database). For new questions pass
# --replay (needs VA_READONLY_DATABASE_URL) or --provider/--model with the key in the environment.
CMD ["va", "serve", "--host", "0.0.0.0", "--port", "8000"]
