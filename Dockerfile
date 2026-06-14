# ─────────────────────────────────────────────────────────────────────────────
# ppi_analyser — production image (multi-stage, slimmed)
# Build:  docker build -t ppi_analyser .
# Run:    see docker-compose.yml
# ─────────────────────────────────────────────────────────────────────────────
# =============================== BUILDER ======================================
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir torch==2.3.1; \
    else \
        pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN STANZA_RESOURCES_DIR=/opt/stanza_resources python - <<'EOF'
import stanza
stanza.download('fr', processors='tokenize,pos,lemma,depparse')
EOF

COPY . .
RUN pip install --no-cache-dir .

# =============================== RUNTIME =======================================
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libpangocairo-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/stanza_resources /root/stanza_resources
COPY --from=builder /app /app

ENV PATH="/opt/venv/bin:$PATH" \
    STANZA_RESOURCES_DIR=/root/stanza_resources \
    PPI_UPLOAD_DIR=/data/uploads \
    PPI_OUTPUT_DIR=/data/output \
    PPI_CACHE_PATH=/data/cache/analysis_cache.json \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data/uploads /data/output /data/cache

EXPOSE 8000

COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
