# PPI Analyser — Docker Deployment Guide

## Architecture inside the container

```
container
├── uvicorn  :8000   ← web UI + REST API  (exposed to the world)
└── stanza   :5000   ← NLP server         (internal only)
```

Two persistent volumes are created automatically by Docker Compose:
- `ppi_data` — uploads, outputs, analysis cache (survives restarts)
- `stanza_models` — French NLP models (avoids re-downloading)

---

## 1. One-time setup on the server

```bash
# clone your repo (or scp the files)
git clone https://github.com/you/ppi_analyser.git
cd ppi_analyser

# copy the env template and fill in your API keys
cp .env.template .env
nano .env          # add MISTRAL_API_KEY, DEEPSEEK_API_KEY, etc.

# copy the Docker files into the repo root
cp path/to/docker-files/* .
```

Apply the 3 small path patches described in `server_env_patch.py`
(or just edit `server.py` directly — they are clearly marked).

---

## 2. Build the image

```bash
docker compose build
# First build takes ~10 min (Torch + Stanza model download).
# Subsequent builds are fast (layers are cached).
```

---

## 3. Run in production

```bash
docker compose up -d          # detached, restarts on reboot
docker compose logs -f        # follow logs
```

The web UI is available at `http://your-server-ip:8000`.

To change the port, edit `HOST_PORT` in `.env`.

---

## 4. Dev mode (hot-reload, bind-mount)

```bash
docker compose --profile dev up
```

Your local source tree is mounted into the container at `/app`.
Every time you save a `.py` file, uvicorn reloads automatically.
You can also `git pull` and the changes are live immediately —
**no rebuild needed**.

---

## 5. Updating the code

```bash
git pull                      # on the server
docker compose restart ppi    # picks up code changes (COPY mode)
# — or —
docker compose --profile dev up   # dev mode: no restart needed
```

If `requirements.txt` changed:

```bash
docker compose build --no-cache
docker compose up -d
```

---

## 6. Accessing results

Results are stored inside the `ppi_data` Docker volume at `/data/output`.
You can copy them out with:

```bash
docker cp ppi_analyser:/data/output ./local-results
```

Or download them directly through the web UI.

---

## 7. Useful commands

```bash
# Check container health
curl http://localhost:8000/health

# Tail live logs
docker compose logs -f ppi

# Open a shell inside the container (debug)
docker exec -it ppi_analyser bash

# Stop everything
docker compose down

# Stop and delete volumes (WARNING: deletes all outputs/cache)
docker compose down -v
```

---

## 8. Reverse proxy (optional, recommended for HTTPS)

If you have Nginx or Caddy on the server, proxy to port 8000:

```nginx
# nginx snippet
server {
    listen 80;
    server_name ppi.yourdomain.fr;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        client_max_body_size 200M;   # for large .xlsx uploads
    }
}
```

Then run `certbot --nginx` for a free TLS certificate.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Stanza server not ready | First boot, model load slow | Wait 60 s, check `docker logs ppi_analyser` |
| `ModuleNotFoundError` | Missing dep in requirements.txt | Add it and rebuild |
| Port 8000 already in use | Another service on the host | Change `HOST_PORT` in `.env` |
| `OLLAMA_HOST` connection refused | Ollama not running or wrong URL | Check `OLLAMA_HOST` in `.env` |
| Results not persisted | Volume not mounted | Check `docker volume ls` |
