# server.py — drop next to core.py
# Run: uvicorn server:app --reload --port 8000
 
import uuid, shutil, threading, traceback, logging, json, io, multiprocessing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os as _os
from dotenv import load_dotenv
load_dotenv()
UPLOADS_DIR = Path(_os.getenv("PPI_UPLOAD_DIR", Path.home() / ".ppi_analyser" / "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict[str, Any]] = {}
job_logs: dict[str, list[str]] = {}
_jobs_lock = threading.Lock()

def _get_client_ip(request: Request) -> str:
    return request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()

# Active process handle — only one at a time
_current_process: multiprocessing.Process | None = None
_current_job_id: str | None = None

# stanza server: required for position detection 

import subprocess
import sys
from pathlib import Path

_stanza_process = None

def start_stanza_server():
    global _stanza_process
    script_path = Path(__file__).parent / "stanza" / "stanza_api.py"
    if not script_path.exists():
        print(f"Warning: {script_path} not found, stanza server not started")
        return
    _stanza_process = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True  # so we can kill process group
    )
    print(f"Started stanza server (PID: {_stanza_process.pid})")

def shutdown_stanza_server():
    global _stanza_process
    if _stanza_process and _stanza_process.poll() is None:
        _stanza_process.terminate()
        try:
            _stanza_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _stanza_process.kill()
        print("Stopped stanza server")




# ── Job store helpers ────────────────────────────────────────────────────────



def _new_job(job_id: str, params: dict, client_ip: str = "") -> dict:
    return {
        "id": job_id, "status": "running", "params": params,
        "created_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
        "error": None, "output_dir": None,
        "tokens_in": 0, "tokens_out": 0, "n_sentences": 0,
        "progress": 0, "phase": "init", "n_total_sentences": 0,
        "client_ip": client_ip,
    }

# ── Constants ────────────────────────────────────────────────────────────────

MODELS_MAPPING = {
    "mistral_large":  "mistral_mistral-large-2411",
    "mistral_medium": "mistral_mistral-medium-2508",
    "deepseek":       "deepseek_deepseek",
    "gemma":          "ollama_gemma3:27b",
    "mistral_local":"ollama_mistral:latest",
    "deepseek_local":"ollama_deepseek-r1:32b",
    "mistral_batch":"mistral_batch_mistral-large-latest"
}

SPEAKER_DETECTION_MODEL = MODELS_MAPPING["deepseek"]

# ── Worker (runs in a separate Process) ─────────────────────────────────────

def _run_job(job_id: str, sentence_file: str, expression: str,
             model_key: str, mode: str, start_sent: int,
             max_sentences: int | str, batch_size: int, n_threads: int,
             use_analysis_cache: bool, selected_props: list[str] | None,
             queue: multiprocessing.Queue):
    """
    Runs entirely in a child Process. Communicates back via queue messages:
      {"type": "log",    "msg": str}
      {"type": "update", "data": dict}   — partial job dict merge
      {"type": "done",   "data": dict}   — final success payload
      {"type": "error",  "msg": str}     — traceback string
    """
    import re as _re
    from ppi_analyser.core import PPIAnalyser
    from ppi_analyser.config import PipelineConfig, AnalysisMode

    # ── Log handler that sends to queue ─────────────────────────────────────
    class QueueLogHandler(logging.Handler):
        def emit(self, record):
            queue.put({"type": "log", "msg": self.format(record)})

    handler = QueueLogHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    root_logger = logging.getLogger("ppi_analyser")
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    def send(type_, **kw):
        queue.put({"type": type_, **kw})

    # ── Validate inputs ──────────────────────────────────────────────────────
    try:
        mode_enum = AnalysisMode(mode)
    except ValueError:
        send("error", msg=f"Mode inconnu : '{mode}'. Valeurs acceptées : {[m.value for m in AnalysisMode]}")
        return

    model_key = "deepseek"  # override for test
    if model_key not in MODELS_MAPPING:
        send("error", msg=f"Modèle inconnu : '{model_key}'. Valeurs acceptées : {list(MODELS_MAPPING)}")
        return

    model_str = MODELS_MAPPING[model_key]

    # ── Output directory ─────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = expression.replace(" ", "_").replace("'", "").replace("/", "_")
    _out_base = Path(_os.getenv("PPI_OUTPUT_DIR", Path.home() / "ppi-analyser-output"))
    out_dir = str(_out_base / f"{slug}_{ts}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    has_preprocessing = (mode == "oral")
    send("update", data={"output_dir": out_dir, "progress": 5, "phase": "init"})

    props_for_pipeline = selected_props if selected_props else None

    config = PipelineConfig(
        models=[model_str],
        expression=expression,
        sentence_file=sentence_file,
        mode=mode_enum,
        output_dir=out_dir,
        start_sent=start_sent,
        max_sentences=max_sentences,
        batch_mode=True,
        batch_size=batch_size,
        n_threads=8, # 8 override for mistral
        use_analysis_cache=use_analysis_cache,
        analysis_cache_path=_os.getenv("PPI_CACHE_PATH",
            str(Path.home() / ".ppi_analyser" / "analysis_cache.json")),
        speaker_detection_model=SPEAKER_DETECTION_MODEL,
        custom_properties=props_for_pipeline,
    )

    try:
        send("update", data={"progress": 10, "phase": "init"})
        analyser = PPIAnalyser(tokenization_mode="nlp")
        first_phase = "preprocessing" if has_preprocessing else "segmentation"
        send("update", data={"progress": 20, "phase": first_phase})

        def _progress(phase, done, n_total):
            # Called by pipeline.py after each sentence/chunk
            # preprocessing maps to 20–45%, analysis to 45–95%
            if n_total == 0:
                return
            ratio = done / n_total
            if phase == "preprocessing":
                pct = int(20 + ratio * 25)   # 20 → 45
            else:
                pct = int(45 + ratio * 50)   # 45 → 95
            send("update", data={
                "phase": phase,
                "progress": pct,
                "n_total_sentences": n_total,
            })

        df, state = analyser.process_sentences(config, progress_callback=_progress)

        if df is None:
            raise RuntimeError("Le pipeline n'a retourné aucun résultat.")

        send("done", data={
            "status": "done",
            "finished_at": datetime.now().isoformat(),
            "tokens_in": state.total_tokens_in,
            "tokens_out": state.total_tokens_out,
            "n_sentences": len(df),
            "progress": 100,
            "phase": "done",
        })

    except Exception:
        send("error", msg=traceback.format_exc())


# ── Queue reader thread (runs in main process) ───────────────────────────────

def _queue_reader(job_id: str, queue: multiprocessing.Queue):
    """
    Reads messages from the child process queue and updates the in-memory
    jobs / job_logs dicts. Exits when it sees a "done" or "error" message,
    or when the queue is empty after the process has exited.
    """
    import queue as _q
    global _current_process, _current_job_id

    while True:
        try:
            msg = queue.get(timeout=1.0)
        except _q.Empty:
            # If the process is gone and queue is empty, we're done
            proc = _current_process
            if proc is not None and not proc.is_alive():
                # Process was killed (terminate()) — mark stopped
                with _jobs_lock:
                    if job_id in jobs and jobs[job_id]["status"] == "running":
                        jobs[job_id].update(
                            status="stopped",
                            finished_at=datetime.now().isoformat(),
                        )
                break
            continue

        t = msg["type"]

        if t == "log":
            with _jobs_lock:
                if job_id in job_logs:
                    job_logs[job_id].append(msg["msg"])


        elif t == "update":
            with _jobs_lock:
                if job_id in jobs:
                    jobs[job_id].update(msg["data"])

        elif t == "done":
            with _jobs_lock:
                if job_id in jobs:
                    jobs[job_id].update(msg["data"])
            break

        elif t == "error":
            with _jobs_lock:
                if job_id in jobs:
                    jobs[job_id].update(
                        status="error",
                        error=msg["msg"],
                        finished_at=datetime.now().isoformat(),
                    )
            break




# ── FastAPI ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_stanza_server()
    yield
    shutdown_stanza_server()

app = FastAPI(title="PPI Analyser API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UI_DIR = Path(__file__).parent / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

@app.get("/", include_in_schema=False)
def root():
    idx = UI_DIR / "index.html"
    return FileResponse(str(idx)) if idx.exists() else {"message": "PPI Analyser API"}


@app.post("/preview")
async def preview_file(
    file: UploadFile = File(...),
    max_rows: int = 2000,
):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "Seuls les fichiers .xlsx sont acceptés.")
    data = await file.read()
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        header = [str(c) if c is not None else "" for c in next(rows_iter, [])]
        rows = []
        total_rows = 0
        for row in rows_iter:
            if all(c is None or str(c).strip() == "" for c in row):
                continue
            total_rows += 1
            if len(rows) < max_rows:
                rows.append([str(c) if c is not None else "" for c in row])
        wb.close()
    except Exception as e:
        raise HTTPException(500, f"Erreur lecture fichier : {e}")
    return {"header": header, "rows": rows, "total_preview": len(rows), "total_rows": total_rows}


@app.post("/analyse", status_code=202)
async def start_analysis(
    request:            Request,
    file:               UploadFile = File(...),
    expression:         str        = Form(...),
    model:              str        = Form("mistral_large"),
    mode:               str        = Form("oral"),
    start_sent:         int        = Form(0),
    max_sentences:      str        = Form("all"),
    batch_size:         int        = Form(5),
    n_threads:          int        = Form(8),
    use_analysis_cache: bool       = Form(True),
    selected_props:     str        = Form("[]"),
):
    global _current_process, _current_job_id

    if not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "Seuls les fichiers .xlsx sont acceptés.")

    # One running job per IP (X-Forwarded-For for reverse-proxy setups)
    client_ip = _get_client_ip(request)
    with _jobs_lock:
        running = [j for j in jobs.values() if j["status"] == "running" and j.get("client_ip") == client_ip]
    if running:
        raise HTTPException(409, f"Un job est déjà en cours pour votre adresse (id: {running[0]['id']}). Attendez qu'il se termine ou arrêtez-le avant d'en lancer un nouveau.")

    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    max_s: int | str = "all"
    if max_sentences.strip().lower() != "all":
        try:
            max_s = int(max_sentences)
        except ValueError:
            raise HTTPException(400, f"max_sentences invalide : '{max_sentences}'")

    props_list: list[str] | None = None
    try:
        parsed = json.loads(selected_props) if selected_props.strip() else []
        props_list = parsed if parsed else None
    except (json.JSONDecodeError, TypeError):
        props_list = None

    job_id = str(uuid.uuid4())
    params = dict(
        expression=expression, model=model, mode=mode,
        original_file=file.filename, start_sent=start_sent,
        max_sentences=max_s, batch_size=batch_size, n_threads=n_threads,
        use_analysis_cache=use_analysis_cache, selected_props=props_list,
    )

    with _jobs_lock:
        jobs[job_id]     = _new_job(job_id, params, client_ip)
        job_logs[job_id] = []

    queue = multiprocessing.Queue()

    proc = multiprocessing.Process(
        target=_run_job,
        args=(job_id, str(dest), expression, model, mode,
              start_sent, max_s, batch_size, n_threads,
              use_analysis_cache, props_list, queue),
        daemon=True,
    )
    proc.start()

    _current_process = proc
    _current_job_id  = job_id

    # Thread that reads the queue and updates in-memory state
    threading.Thread(target=_queue_reader, args=(job_id, queue), daemon=True).start()

    return {"job_id": job_id}


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str, request: Request):
    global _current_process, _current_job_id
    client_ip = _get_client_ip(request)

    with _jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job introuvable.")
        if jobs[job_id].get("client_ip") != client_ip:
            raise HTTPException(403, "Ce job ne vous appartient pas.")
        if jobs[job_id]["status"] != "running":
            raise HTTPException(400, f"Le job n'est pas en cours (statut : {jobs[job_id]['status']}).")

    if _current_job_id == job_id and _current_process is not None:
        _current_process.terminate()
        _current_process = None
        _current_job_id  = None
        with _jobs_lock:
            jobs[job_id].update(
                status="stopped",
                finished_at=datetime.now().isoformat(),
            )

    return {"stopped": job_id}


@app.get("/jobs")
def list_jobs(request: Request):
    client_ip = _get_client_ip(request)
    with _jobs_lock:
        return sorted(
            [j for j in jobs.values() if j.get("client_ip") == client_ip],
            key=lambda j: j["created_at"], reverse=True
        )

@app.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    client_ip = _get_client_ip(request)
    with _jobs_lock:
        j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job introuvable.")
    if j.get("client_ip") != client_ip:
        raise HTTPException(403, "Ce job ne vous appartient pas.")
    return j

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, request: Request):
    client_ip = _get_client_ip(request)
    with _jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job introuvable.")
        if jobs[job_id].get("client_ip") != client_ip:
            raise HTTPException(403, "Ce job ne vous appartient pas.")
        if jobs[job_id]["status"] == "running":
            raise HTTPException(400, "Impossible de supprimer un job en cours. Arrêtez-le d'abord.")
        del jobs[job_id]
        job_logs.pop(job_id, None)
    return {"deleted": job_id}

@app.get("/jobs/{job_id}/logs")
def get_logs(job_id: str, request: Request, since: int = 0):
    client_ip = _get_client_ip(request)
    with _jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job introuvable.")
        if jobs[job_id].get("client_ip") != client_ip:
            raise HTTPException(403, "Ce job ne vous appartient pas.")
        lines = job_logs.get(job_id, [])[since:]
    return {"lines": lines, "total": since + len(lines)}

@app.get("/results/{job_id}")
def list_results(job_id: str, request: Request):
    client_ip = _get_client_ip(request)
    with _jobs_lock:
        j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job introuvable.")
    if j.get("client_ip") != client_ip:
        raise HTTPException(403, "Ce job ne vous appartient pas.")
    if j["status"] != "done":
        raise HTTPException(425, "Le job n'est pas encore terminé.")
    out = Path(j["output_dir"])
    return {
        "output_dir": str(out),
        "files": [
            {"name": f.name, "path": str(f),
             "size_kb": round(f.stat().st_size / 1024, 1), "suffix": f.suffix}
            for f in sorted(out.iterdir()) if f.is_file()
        ],
    }

@app.get("/download/{job_id}/{filename}")
def download_file(job_id: str, filename: str, request: Request):
    client_ip = _get_client_ip(request)
    with _jobs_lock:
        j = jobs.get(job_id)
    if not j or j["status"] != "done":
        raise HTTPException(404, "Introuvable ou job non terminé.")
    if j.get("client_ip") != client_ip:
        raise HTTPException(403, "Ce job ne vous appartient pas.")
    p = Path(j["output_dir"]) / filename
    if not p.exists():
        raise HTTPException(404, "Fichier introuvable.")
    return FileResponse(str(p), filename=filename)

@app.get("/health")
def health():
    return {"status": "ok", "jobs_in_memory": len(jobs)}


# ── Admin endpoints ──────────────────────────────────────────────────────────

_ADMIN_SECRET = _os.getenv("ADMIN_SECRET", "changeme")

def _check_admin(x_admin_secret: str = None):
    if x_admin_secret != _ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")

from fastapi import Header as _Header

@app.post("/admin/set-env")
def admin_set_env(
    payload: dict,
    x_admin_secret: str = _Header(None),
):
    _check_admin(x_admin_secret)
    key = payload.get("key", "").strip()
    value = payload.get("value", "")
    if not key:
        raise HTTPException(400, "Missing key")
    _os.environ[key] = value
    return {"set": key}


@app.post("/admin/pull")
def admin_pull(x_admin_secret: str = _Header(None)):
    _check_admin(x_admin_secret)
    import urllib.request, zipfile, io as _io
    GITHUB_REPO = _os.getenv("GITHUB_REPO", "YoussefZeroual/ppi-analyser")
    GITHUB_BRANCH = _os.getenv("GITHUB_BRANCH", "main")
    url = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"
    cwd = Path(__file__).parent.parent
    try:
        with urllib.request.urlopen(url) as r:
            z = zipfile.ZipFile(_io.BytesIO(r.read()))
        updated = 0
        for member in z.namelist():
            parts = Path(member).parts
            if len(parts) < 2:
                continue
            target = cwd / Path(*parts[1:])
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                data = z.read(member)
                if not target.exists() or target.read_bytes() != data:
                    target.write_bytes(data)
                    updated += 1
        return {"status": "ok", "files_updated": updated, "repo": GITHUB_REPO, "branch": GITHUB_BRANCH}
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
