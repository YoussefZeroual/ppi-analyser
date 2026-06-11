# models/mistral.py

import json
import time
import logging
from pathlib import Path
from ppi_analyser.models.base import LLMProvider

logger = logging.getLogger(__name__)


class MistralProvider(LLMProvider):

    def __init__(self, submodel: str, api_key: str):
        self.submodel = submodel
        self.api_key  = api_key

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from mistralai.client import Mistral
        with Mistral(api_key=self.api_key) as client:
            response = client.chat.complete(
                model=self.submodel,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )
        return response.choices[0].message.content


class MistralBatchProvider:
    """
    Wraps Mistral's asynchronous Batch API (50% cheaper than standard).
    Uses inline requests — no file upload needed.

    Typical flow:
        provider = MistralBatchProvider(submodel, api_key, output_dir)
        job_id   = provider.submit(requests)          # list of {custom_id, system, user}
        results  = provider.poll_or_save(job_id)      # blocks until done or timeout
        # if None returned: re-run script, it will resume automatically
    """

    POLL_INTERVAL = 30      # seconds between status checks
    TIMEOUT       = 3600    # seconds before saving job_id and giving up (1 hour)

    def __init__(self, submodel: str, api_key: str, output_dir: str):
        self.submodel   = submodel
        self.api_key    = api_key
        self.output_dir = Path(output_dir)

    def _client(self):
        from mistralai.client import Mistral
        return Mistral(api_key=self.api_key)

    def _job_state_path(self) -> Path:
        return self.output_dir / "mistral_batch_job.json"

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(self, requests: list[dict]) -> str:
        """
        Submit a batch job using inline requests (no file upload needed).
        requests: list of {"custom_id": str, "system": str, "user": str}
        Returns the job_id string.
        """
        inline_data = [
            {
                "custom_id": req["custom_id"],
                "body": {
                "temperature": 0,
                    "messages": [
                        {"role": "system", "content": req["system"]},
                        {"role": "user",   "content": req["user"]},
                    ]
                }
            }
            for req in requests
        ]

        with self._client() as client:
            job = client.batch.jobs.create(
                requests=inline_data,
                model=self.submodel,
                endpoint="/v1/chat/completions",
                metadata={"source": "ppi_analyser"},
            )

        logger.info("Mistral batch: created job %s (%d requests)", job.id, len(requests))
        return job.id

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def poll_or_save(self, job_id: str, preprocessed_json: str | None = None) -> dict | None:
        """
        Poll until job completes or TIMEOUT is reached.
        Returns dict {custom_id: result_str} on success, or None if timed out.
        On timeout, saves job state to output_dir for resume.
        """
        start = time.time()

        while True:
            with self._client() as client:
                job = client.batch.jobs.get(job_id=job_id)

            logger.info(
                "Mistral batch job %s — status: %s (%s/%s done)",
                job_id, job.status,
                getattr(job, 'succeeded_requests', '?'),
                getattr(job, 'total_requests', '?'),
            )

            if job.status == "SUCCESS":
                return self._download_results(job_id, job.output_file)

            if job.status in ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
                raise RuntimeError(f"Mistral batch job {job_id} ended with status: {job.status}")

            if time.time() - start > self.TIMEOUT:
                logger.warning(
                    "Mistral batch job %s not done after %ds — saving job ID to resume later",
                    job_id, self.TIMEOUT
                )
                self._save_job_state(job_id, preprocessed_json)
                return None

            time.sleep(self.POLL_INTERVAL)

    def resume(self, job_id: str, preprocessed_json: str | None = None) -> dict | None:
        """Resume polling a previously saved job."""
        with self._client() as client:
            job = client.batch.jobs.get(job_id=job_id)

        if job.status == "SUCCESS":
            return self._download_results(job_id, job.output_file)

        if job.status in ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
            raise RuntimeError(f"Mistral batch job {job_id} ended with status: {job.status}")

        logger.info("Mistral batch job %s still running (status: %s) — continuing to poll", job_id, job.status)
        return self.poll_or_save(job_id, preprocessed_json)

    # ------------------------------------------------------------------
    # Download + parse
    # ------------------------------------------------------------------

    def _download_results(self, job_id: str, output_file_id: str) -> dict:
        """Download result JSONL and return {custom_id: result_str}."""
        with self._client() as client:
            content = client.files.download(file_id=output_file_id).read().decode("utf-8")

        # Save raw output for debugging
        raw_path = self.output_dir / f"mistral_batch_{job_id}_raw.jsonl"
        raw_path.write_text(content, encoding="utf-8")
        logger.info("Mistral batch: raw results saved to %s", raw_path)

        results = {}
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                obj       = json.loads(line)
                custom_id = obj["custom_id"]
                message   = obj["response"]["body"]["choices"][0]["message"]["content"]
                results[custom_id] = message
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning("Mistral batch: could not parse result line: %s — %s", line[:80], e)

        logger.info("Mistral batch job %s completed: %d results", job_id, len(results))
        return results

    # ------------------------------------------------------------------
    # Job state persistence
    # ------------------------------------------------------------------

    def _save_job_state(self, job_id: str, preprocessed_json: str | None) -> None:
        state = {"job_id": job_id, "preprocessed_json": preprocessed_json}
        self._job_state_path().write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Mistral batch: job state saved to %s", self._job_state_path())
        print(f"\nMistral batch job not yet complete. Job ID saved to:\n  {self._job_state_path()}")
        print("Re-run with the same output_dir and batch_mode=True to resume.\n")

    def load_job_state(self) -> dict | None:
        path = self._job_state_path()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def clear_job_state(self) -> None:
        path = self._job_state_path()
        if path.exists():
            path.unlink()
            logger.info("Mistral batch: job state cleared")
