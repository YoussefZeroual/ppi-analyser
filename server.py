# server.py — drop next to core.py
# Run: uvicorn server:app --reload --port 8000

import uuid, shutil, threading, traceback, logging, json, io
from datetime import datetime
from pathlib import Path
from typing import Any

import openpyxl

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

UPLOADS_DIR = Path.home() / ".ppi_analyser" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict[str, Any]] = {}
job_logs: dict[str, list[str]] = {}
job_stop_events: dict[str, threading.Event] = {}
_jobs_lock = threading.Lock()

# ── Per-job log handler ──────────────────────────────────────────────────────

class JobLogHandler(logging.Handler):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        with _jobs_lock:
            if self.job_id in job_logs:
                job_logs[self.job_id].append(msg)

# ── Job store helpers ────────────────────────────────────────────────────────

def _new_job(job_id: str, params: dict) -> dict:
    return {
        "id": job_id, "status": "running", "params": params,
        "created_at": datetime.now().isoformat(), "finished_at": None,
        "error": None, "output_dir": None,
        "tokens_in": 0, "tokens_out": 0, "n_sentences": 0,
        "progress": 0,
    }

# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="PPI Analyser API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UI_DIR = Path(__file__).parent / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

@app.get("/", include_in_schema=False)
def root():
    idx = UI_DIR / "index.html"
    return FileResponse(str(idx)) if idx.exists() else {"message": "PPI Analyser API"}

# ── Constants ────────────────────────────────────────────────────────────────

# Maps the short UI key → full provider_model string used by the pipeline
MODELS_MAPPING = {
    "mistral_large":  "mistral_mistral-large-2411",
    "mistral_medium": "mistral_mistral-medium-2508",
    "deepseek":       "deepseek_deepseek",
    "gemma":          "ollama_gemma3:27b",
}

# Speaker detection model — same as CLI
SPEAKER_DETECTION_MODEL = "mistral_mistral-large-2411"

# ── Background worker ────────────────────────────────────────────────────────

def _run_job(job_id: str, sentence_file: str, expression: str,
             model_key: str, mode: str, start_sent: int,
             max_sentences: int | str, batch_size: int, n_threads: int,
             use_analysis_cache: bool, selected_props: list[str] | None,
             stop_event: threading.Event):

    from ppi_analyser.core import PPIAnalyser
    from ppi_analyser.config import PipelineConfig, AnalysisMode

    # Attach a per-job log handler so every ppi_analyser log goes to the UI
    handler = JobLogHandler(job_id)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    root_logger = logging.getLogger("ppi_analyser")
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    def fail(msg: str):
        with _jobs_lock:
            jobs[job_id].update(status="error", error=msg,
                                finished_at=datetime.now().isoformat())
        root_logger.removeHandler(handler)

    # Early-exit if already stopped before we even start
    if stop_event.is_set():
        with _jobs_lock:
            jobs[job_id].update(status="stopped", finished_at=datetime.now().isoformat())
        root_logger.removeHandler(handler)
        return

    # ── Validate inputs ──────────────────────────────────────────────────────
    try:
        mode_enum = AnalysisMode(mode)
    except ValueError:
        return fail(f"Mode inconnu : '{mode}'. Valeurs acceptées : {[m.value for m in AnalysisMode]}")
    model_key = "gemma" # override for test purpose
    if model_key not in MODELS_MAPPING:
        return fail(f"Modèle inconnu : '{model_key}'. Valeurs acceptées : {list(MODELS_MAPPING)}")

    model_str = MODELS_MAPPING[model_key]

    # ── Output directory ─────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = expression.replace(" ", "_").replace("'", "").replace("/", "_")
    out_dir = str(Path.home() / "ppi-analyser-output" / f"{slug}_{ts}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with _jobs_lock:
        jobs[job_id]["output_dir"] = out_dir
        jobs[job_id]["progress"]   = 5

    # ── PipelineConfig — mirrors test.py exactly ─────────────────────────────
    #   The `properties` field controls which props are analysed (None = all).
    #   selected_props=[] (empty list from UI) means "none selected" which is
    #   invalid; we treat it as None (= all) to be safe.
    props_for_pipeline = selected_props if selected_props else None

    config = PipelineConfig(
        models=[model_str],
        expression=expression,
        sentence_file=sentence_file,
        mode=mode_enum,
        output_dir=out_dir,
        start_sent=start_sent,
        max_sentences=max_sentences,   # "all" or int, exactly as CLI
        batch_mode=True,
        batch_size=batch_size,
        n_threads=n_threads,
        use_analysis_cache=use_analysis_cache,
        analysis_cache_path=str(Path.home() / ".ppi_analyser" / "analysis_cache.json"),
        speaker_detection_model=SPEAKER_DETECTION_MODEL,
        custom_properties=props_for_pipeline,  # None → all props; list → only those
    )

    try:
        with _jobs_lock:
            jobs[job_id]["progress"] = 10

        analyser = PPIAnalyser(tokenization_mode="nlp")

        with _jobs_lock:
            jobs[job_id]["progress"] = 20

        # ── Run process_sentences in a sub-thread so we can poll stop_event ──
        result     = [None, None]
        exc_holder = [None]

        def _inner():
            try:
                result[0], result[1] = analyser.process_sentences(config)
            except Exception as exc:
                exc_holder[0] = exc

        inner = threading.Thread(target=_inner, daemon=True)
        inner.start()

        # Poll every 500 ms; update progress heuristically while running
        tick = 0
        while inner.is_alive():
            inner.join(timeout=0.5)
            tick += 1
            # Slowly advance the progress bar from 20 → 90 while running
            # (real progress would require pipeline callbacks)
            if inner.is_alive():
                with _jobs_lock:
                    cur = jobs[job_id].get("progress", 20)
                    if cur < 90:
                        jobs[job_id]["progress"] = min(90, cur + 1)
            if stop_event.is_set():
                root_logger.warning("Arrêt demandé — attente de la fin du batch en cours…")
                inner.join()
                break

        if exc_holder[0]:
            raise exc_holder[0]

        df, state = result[0], result[1]
        if df is None:
            raise RuntimeError("Le pipeline n'a retourné aucun résultat.")

        if stop_event.is_set():
            with _jobs_lock:
                jobs[job_id].update(
                    status="stopped",
                    finished_at=datetime.now().isoformat(),
                    tokens_in=getattr(state, "total_tokens_in", 0),
                    tokens_out=getattr(state, "total_tokens_out", 0),
                    n_sentences=len(df),
                    progress=jobs[job_id].get("progress", 0),
                )
        else:
            with _jobs_lock:
                jobs[job_id].update(
                    status="done",
                    finished_at=datetime.now().isoformat(),
                    tokens_in=state.total_tokens_in,
                    tokens_out=state.total_tokens_out,
                    n_sentences=len(df),
                    progress=100,
                )

    except Exception:
        fail(traceback.format_exc())
    finally:
        root_logger.removeHandler(handler)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/preview")
async def preview_file(
    file: UploadFile = File(...),
    max_rows: int = 50,
):
    """Return the first max_rows rows of an xlsx file as JSON (header + rows)."""
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "Seuls les fichiers .xlsx sont acceptés.")
    data = await file.read()
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        header = [str(c) if c is not None else "" for c in next(rows_iter, [])]
        rows = []
        for i, row in enumerate(rows_iter):
            if i >= max_rows:
                break
            rows.append([str(c) if c is not None else "" for c in row])
        wb.close()
    except Exception as e:
        raise HTTPException(500, f"Erreur lecture fichier : {e}")
    return {"header": header, "rows": rows, "total_preview": len(rows)}


@app.post("/analyse", status_code=202)
async def start_analysis(
    file:               UploadFile = File(...),
    expression:         str        = Form(...),
    model:              str        = Form("mistral_large"),
    mode:               str        = Form("oral"),
    start_sent:         int        = Form(0),
    max_sentences:      str        = Form("all"),
    batch_size:         int        = Form(5),
    n_threads:          int        = Form(8),
    # FastAPI parses "true"/"false" strings from FormData correctly as bool
    use_analysis_cache: bool       = Form(True),
    # JSON-encoded list of property keys, e.g. '["Position","Modalite"]'
    # Empty string or "[]" → None (all properties)
    selected_props:     str        = Form("[]"),
):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "Seuls les fichiers .xlsx sont acceptés.")

    # Save uploaded file
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # Parse max_sentences
    max_s: int | str = "all"
    if max_sentences.strip().lower() != "all":
        try:
            max_s = int(max_sentences)
        except ValueError:
            raise HTTPException(400, f"max_sentences invalide : '{max_sentences}'")

    # Parse selected_props JSON list
    props_list: list[str] | None = None
    try:
        parsed = json.loads(selected_props) if selected_props.strip() else []
        props_list = parsed if parsed else None  # empty list → None = all props
    except (json.JSONDecodeError, TypeError):
        props_list = None

    job_id     = str(uuid.uuid4())
    stop_event = threading.Event()

    params = dict(
        expression=expression,
        model=model,
        mode=mode,
        original_file=file.filename,
        start_sent=start_sent,
        max_sentences=max_s,
        batch_size=batch_size,
        n_threads=n_threads,
        use_analysis_cache=use_analysis_cache,
        # Store the human-readable list for the UI recap
        selected_props=props_list,
    )

    with _jobs_lock:
        jobs[job_id]            = _new_job(job_id, params)
        job_logs[job_id]        = []
        job_stop_events[job_id] = stop_event

    threading.Thread(
        target=_run_job,
        args=(job_id, str(dest), expression, model, mode,
              start_sent, max_s, batch_size, n_threads,
              use_analysis_cache, props_list, stop_event),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    with _jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job introuvable.")
        if jobs[job_id]["status"] != "running":
            raise HTTPException(400, f"Le job n'est pas en cours (statut : {jobs[job_id]['status']}).")
        event = job_stop_events.get(job_id)
    if event:
        event.set()
    return {"stopped": job_id}


@app.get("/jobs")
def list_jobs():
    with _jobs_lock:
        return sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job introuvable.")
    return j

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    with _jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job introuvable.")
        event = job_stop_events.get(job_id)
        if event:
            event.set()   # signal stop if still running
        del jobs[job_id]
        job_logs.pop(job_id, None)
        job_stop_events.pop(job_id, None)
    return {"deleted": job_id}

@app.get("/jobs/{job_id}/logs")
def get_logs(job_id: str, since: int = 0):
    with _jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job introuvable.")
        lines = job_logs.get(job_id, [])[since:]
    return {"lines": lines, "total": since + len(lines)}

@app.get("/results/{job_id}")
def list_results(job_id: str):
    with _jobs_lock:
        j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job introuvable.")
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
def download_file(job_id: str, filename: str):
    with _jobs_lock:
        j = jobs.get(job_id)
    if not j or j["status"] != "done":
        raise HTTPException(404, "Introuvable ou job non terminé.")
    p = Path(j["output_dir"]) / filename
    if not p.exists():
        raise HTTPException(404, "Fichier introuvable.")
    return FileResponse(str(p), filename=filename)

@app.get("/health")
def health():
    return {"status": "ok", "jobs_in_memory": len(jobs)}
