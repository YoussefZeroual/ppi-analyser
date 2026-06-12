# ─────────────────────────────────────────────────────────────────────────────
# ppi_analyser — production image (multi-stage, slimmed)
# Build:  docker build -t ppi_analyser .
# Run:    see docker-compose.yml
# ─────────────────────────────────────────────────────────────────────────────

# =============================== BUILDER ======================================
FROM python:3.11-slim-bookworm AS builder

# Build-time only deps (compilers, headers) — none of this ends up in final image
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install into a venv so we can copy it cleanly to the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install torch CPU-only explicitly FIRST and pinned, so the resolver doesn't
# later pull in CUDA wheels (nvidia-cu13-*, triton, cuda-toolkit, etc.) when
# resolving other packages' torch dependency from the default PyPI index.
RUN pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Stanza French models (drop depparse if you don't need it — it's the
# largest piece, mostly pretrained word embeddings)
RUN STANZA_RESOURCES_DIR=/opt/stanza_resources python - <<'EOF'
import stanza
stanza.download('fr', processors='tokenize,pos,lemma,depparse')
EOF

# Install the package itself (non-editable in prod — editable mode pulls in
# extra build metadata and isn't needed once code is COPY'd)
COPY . .
RUN pip install --no-cache-dir .


# =============================== RUNTIME =======================================
FROM python:3.11-slim-bookworm AS runtime

# Only runtime system deps (WeasyPrint needs these; no -dev / compiler packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libpangocairo-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the prebuilt venv (includes all pip packages) and app code
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

# 8000 → FastAPI/uvicorn (web UI + REST API)
# 5000 → Stanza API server (internal, not needed outside the container)
EXPOSE 8000

COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
