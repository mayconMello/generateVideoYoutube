# src/integrations/editly_api.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional
import requests
from pathlib import Path
import copy


def _api_base() -> str:
    base = os.getenv("EDITLY_API_URL", "http://localhost:3535").rstrip("/")
    return base


def wait_for_health(timeout_sec: float = 5.0, interval_sec: float = 0.5) -> bool:
    """
    Probe Editly API health endpoint until it responds or timeout.
    Env: EDITLY_API_URL (default http://localhost:3535)
    """
    base = _api_base()
    deadline = time.time() + max(0.1, float(timeout_sec))
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/health", timeout=3)
            if r.ok:
                return True
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(max(0.05, float(interval_sec)))
    return False


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json() if r.content else {}
    except json.JSONDecodeError:
        return {}


def _map_host_to_container_path(path_str: str) -> str:
    """
    Map an absolute host path into the container mount root so Editly inside Docker
    can access files. When EDITLY_MOUNT_ROOT is set (e.g. '/work') and the current
    working directory is bind-mounted there, replace the host project root prefix
    with that container path.

    If mapping conditions are not met, return the original path.
    """
    try:
        if not isinstance(path_str, str) or not path_str:
            return path_str
        # If already a container path, keep it
        if path_str.startswith("/work/"):
            return path_str
        # Default to '/work' to match docker-compose volume '.:/work'
        mount_root = os.getenv("EDITLY_MOUNT_ROOT") or "/work"
        host_root_env = os.getenv("EDITLY_HOST_ROOT")
        host_root = Path(host_root_env).resolve() if host_root_env else Path.cwd().resolve()
        p_abs = Path(path_str).resolve()
        try:
            rel = p_abs.relative_to(host_root)
        except Exception:
            # Not under project root; leave as is
            return path_str
        return str(Path(mount_root).joinpath(rel))
    except Exception:
        return path_str


def _transform_paths_for_container(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a copy of the config with file paths rewritten for the container.
    Rewrites keys: outPath, audioFilePath, all layer.path occurrences.
    """
    out = copy.deepcopy(cfg)
    try:
        if isinstance(out.get("outPath"), str):
            out["outPath"] = _map_host_to_container_path(out["outPath"])
        if isinstance(out.get("audioFilePath"), str):
            out["audioFilePath"] = _map_host_to_container_path(out["audioFilePath"])
        clips = out.get("clips") or []
        if isinstance(clips, list):
            for c in clips:
                if not isinstance(c, dict):
                    continue
                layers = c.get("layers")
                if isinstance(layers, list):
                    for layer in layers:
                        if isinstance(layer, dict) and isinstance(layer.get("path"), str):
                            layer["path"] = _map_host_to_container_path(layer["path"])
    except Exception:
        return out
    return out


def render_with_editly_api(
    config: Dict[str, Any],
    *,
    timeout_sec: float = 1800.0,
    poll_interval_sec: float = 1.5,
) -> bool:
    """
    Submit a render job to Editly API and block until completion (or timeout).

    Accepts either:
      - Synchronous API: POST /render returns {"ok": true}
      - Async API: POST /render returns {"job_id": "..."} and exposes /jobs/{id}

    Returns True on success, False otherwise.
    """
    base = _api_base()
    start = time.time()

    try:
        payload = _transform_paths_for_container(config)
        resp = _post_json(f"{base}/render", payload, timeout=15)
    except requests.RequestException:
        return False

    # Synchronous success path
    if isinstance(resp, dict) and resp.get("ok") is True:
        return True

    # Asynchronous job path
    job_id = resp.get("job_id") if isinstance(resp, dict) else None
    if not isinstance(job_id, str) or not job_id:
        # Some servers return {"status":"queued","id":"..."}
        job_id = resp.get("id") if isinstance(resp, dict) else None
    if not isinstance(job_id, str) or not job_id:
        return False

    # Poll job status
    status_url = f"{base}/jobs/{job_id}"
    while (time.time() - start) < timeout_sec:
        try:
            s = requests.get(status_url, timeout=10)
            if not s.ok:
                time.sleep(poll_interval_sec)
                continue
            data = s.json() if s.content else {}
        except requests.RequestException:
            time.sleep(poll_interval_sec)
            continue

        status = (data.get("status") or "").lower()
        if status in {"done", "completed", "success"}:
            return True
        if status in {"failed", "error", "canceled", "cancelled"}:
            return False
        time.sleep(poll_interval_sec)

    return False
