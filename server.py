# server.py — drop next to core.py
# Run: uvicorn server:app --reload --port 8000

import uuid, shutil, threading, traceback, logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

UPLOADS_DIR = Path.home() / ".ppi_analyser" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict[str, Any]] = {}
job_logs: dict[str, list[str]] = {}      # job_id -> list of log lines
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
        "progress": 0,   # 0–100
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

MODELS_MAPPING = {
    "deepseek":       "deepseek_deepseek",
    "mistral":        "mistral_batch_mistral-medium-latest",
    "gemma":          "ollama_gemma3:27b",
    "mistral_medium": "mistral_mistral-medium-2508",
    "mistral_large":"mistral_mistral-large-latest"
}
MODES = ["oral", "écrit", "écrit_ia", "écrit_test"]

# ── Background worker ────────────────────────────────────────────────────────

def _run_job(job_id, sentence_file, expression, model, mode,
             start_sent, max_sentences, batch_size, n_threads, use_analysis_cache):
    from ppi_analyser.core import PPIAnalyser
    from ppi_analyser.config import PipelineConfig, AnalysisMode

    # Attach log handler so all ppi_analyser logs go to this job
    handler = JobLogHandler(job_id)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                            datefmt="%H:%M:%S"))
    root_logger = logging.getLogger("ppi_analyser")
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    def fail(msg):
        with _jobs_lock:
            jobs[job_id].update(status="error", error=msg,
                                finished_at=datetime.now().isoformat())
        root_logger.removeHandler(handler)

    try:
        mode_enum = AnalysisMode(mode)
    except ValueError:
        return fail(f"Unknown mode '{mode}'")
    if model not in MODELS_MAPPING:
        return fail(f"Unknown model '{model}'")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = expression.replace(" ", "_").replace("'", "")
    out_dir = str(Path.home() / "ppi-analyser-output" / f"{slug}_{ts}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with _jobs_lock:
        jobs[job_id]["output_dir"] = out_dir
        jobs[job_id]["progress"] = 5

    config = PipelineConfig(
        models=[MODELS_MAPPING[model]], expression=expression,
        sentence_file=sentence_file, mode=mode_enum, output_dir=out_dir,
        start_sent=start_sent, max_sentences=max_sentences,
        batch_mode=True, batch_size=batch_size, n_threads=n_threads,
        use_analysis_cache=use_analysis_cache,
        analysis_cache_path=str(Path.home() / ".ppi_analyser" / "analysis_cache.json"),
        speaker_detection_model="mistral_mistral-large-latest",
    )
    try:
        with _jobs_lock:
            jobs[job_id]["progress"] = 10
        analyser = PPIAnalyser(tokenization_mode="nlp")
        with _jobs_lock:
            jobs[job_id]["progress"] = 20

        df, state = analyser.process_sentences(config)

        with _jobs_lock:
            jobs[job_id].update(
                status="done", finished_at=datetime.now().isoformat(),
                tokens_in=state.total_tokens_in, tokens_out=state.total_tokens_out,
                n_sentences=len(df), progress=100,
            )
    except Exception:
        fail(traceback.format_exc())
    finally:
        root_logger.removeHandler(handler)

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/analyse", status_code=202)
async def start_analysis(
    file:               UploadFile = File(...),
    expression:         str        = Form(...),
    model:              str        = Form("deepseek"),
    mode:               str        = Form("oral"),
    start_sent:         int        = Form(0),
    max_sentences:      str        = Form("all"),
    batch_size:         int        = Form(5),
    n_threads:          int        = Form(8),
    use_analysis_cache: bool       = Form(True),
):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "Only .xlsx files accepted")
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    max_s = "all" if max_sentences.strip().lower() == "all" else int(max_sentences)
    job_id = str(uuid.uuid4())
    params = dict(expression=expression, model=model, mode=mode,
                  original_file=file.filename, start_sent=start_sent,
                  max_sentences=max_s, batch_size=batch_size,
                  n_threads=n_threads, use_analysis_cache=use_analysis_cache)
    with _jobs_lock:
        jobs[job_id]     = _new_job(job_id, params)
        job_logs[job_id] = []

    threading.Thread(
        target=_run_job,
        args=(job_id, str(dest), expression, model, mode,
              start_sent, max_s, batch_size, n_threads, use_analysis_cache),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/jobs")
def list_jobs():
    with _jobs_lock:
        return sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        j = jobs.get(job_id)
    if not j: raise HTTPException(404, "Job not found")
    return j

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    with _jobs_lock:
        if job_id not in jobs: raise HTTPException(404, "Job not found")
        del jobs[job_id]
        job_logs.pop(job_id, None)
    return {"deleted": job_id}

@app.get("/jobs/{job_id}/logs")
def get_logs(job_id: str, since: int = 0):
    """Return log lines since index `since`."""
    with _jobs_lock:
        if job_id not in jobs: raise HTTPException(404, "Job not found")
        lines = job_logs.get(job_id, [])[since:]
    return {"lines": lines, "total": since + len(lines)}

@app.get("/results/{job_id}")
def list_results(job_id: str):
    with _jobs_lock:
        j = jobs.get(job_id)
    if not j: raise HTTPException(404, "Job not found")
    if j["status"] != "done": raise HTTPException(425, "Job not finished yet")
    out = Path(j["output_dir"])
    return {"output_dir": str(out), "files": [
        {"name": f.name, "path": str(f),
         "size_kb": round(f.stat().st_size/1024, 1), "suffix": f.suffix}
        for f in sorted(out.iterdir()) if f.is_file()
    ]}

@app.get("/download/{job_id}/{filename}")
def download_file(job_id: str, filename: str):
    with _jobs_lock:
        j = jobs.get(job_id)
    if not j or j["status"] != "done": raise HTTPException(404, "Not found or not done")
    p = Path(j["output_dir"]) / filename
    if not p.exists(): raise HTTPException(404, "File not found")
    return FileResponse(str(p), filename=filename)

@app.get("/health")
def health():
    return {"status": "ok", "jobs_in_memory": len(jobs)}
