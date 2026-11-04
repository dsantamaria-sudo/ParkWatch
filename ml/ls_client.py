"""ParkWatch — Label Studio export, tasks and labels."""
from __future__ import annotations

import json
import urllib.request

from azure_utils import retry

FROM_NAME = "spot"  # label_studio/labeling_config.xml


def _http_json(method: str, url: str, body=None, headers: dict | None = None, timeout: int = 600):
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def ls_headers(ls_url: str, token: str) -> dict:
    """Legacy token -> 'Token <t>'; Personal Access Token (refresh JWT) -> exchanged for an access token."""
    if token.startswith("eyJ") and token.count(".") == 2:
        access = retry(
            lambda: _http_json("POST", f"{ls_url}/api/token/refresh", {"refresh": token})["access"],
            "Label Studio token",
        )
        return {"Authorization": f"Bearer {access}"}
    return {"Authorization": f"Token {token}"}


def export_tasks(ls_url: str, token: str, project: int) -> list[dict]:
    headers = ls_headers(ls_url, token)
    return retry(
        lambda: _http_json("GET", f"{ls_url}/api/projects/{project}/export?exportType=JSON", headers=headers),
        "Label Studio export",
    )


TASKS_PAGE = 500
IMPORT_BATCH = 100


def list_tasks(ls_url: str, headers: dict, project: int) -> list[dict]:
    """Every task of the project with full annotations and predictions (fields=all)."""
    import urllib.error

    tasks, page = [], 1
    while True:
        url = f"{ls_url}/api/tasks?project={project}&page={page}&page_size={TASKS_PAGE}&fields=all"
        try:
            resp = retry(lambda: _http_json("GET", url, headers=headers), f"tasks page {page}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # page out of range
                break
            raise
        batch = resp.get("tasks", []) if isinstance(resp, dict) else resp
        tasks.extend(batch)
        if len(batch) < TASKS_PAGE:
            break
        page += 1
    return tasks


def is_annotated(task: dict) -> bool:
    """Any annotation, including cancelled/skipped ones: those tasks are never touched."""
    return bool(task.get("annotations")) or task.get("total_annotations", 0) > 0 or task.get("cancelled_annotations", 0) > 0


def get_prediction(ls_url: str, headers: dict, prediction) -> dict:
    """fields=all may return the full prediction or only its id."""
    if isinstance(prediction, dict) and "result" in prediction:
        return prediction
    pid = prediction["id"] if isinstance(prediction, dict) else prediction
    return retry(lambda: _http_json("GET", f"{ls_url}/api/predictions/{pid}", headers=headers), f"prediction {pid}")


def update_prediction(ls_url: str, headers: dict, prediction_id: int, result: list[dict], score: float) -> None:
    retry(
        lambda: _http_json("PATCH", f"{ls_url}/api/predictions/{prediction_id}",
                           {"result": result, "score": score}, headers),
        f"prediction {prediction_id}",
    )


def import_tasks(ls_url: str, headers: dict, project: int, tasks: list[dict]) -> None:
    for i in range(0, len(tasks), IMPORT_BATCH):
        batch = tasks[i : i + IMPORT_BATCH]
        retry(lambda: _http_json("POST", f"{ls_url}/api/projects/{project}/import", batch, headers), "import tasks")
        print(f"  imported {len(batch)} tasks (batch {i // IMPORT_BATCH + 1})")


def latest_labels(task: dict, spot_ids: set[str]) -> dict[str, str] | None:
    """Labels per spot from the latest non-cancelled annotation (None if the task has none)."""
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled") and a.get("result")]
    if not anns:
        return None
    ann = max(anns, key=lambda a: a["updated_at"])
    return {
        r["id"]: r["value"]["polygonlabels"][0]
        for r in ann["result"]
        if r.get("type") == "polygonlabels" and r.get("from_name") == FROM_NAME and r["id"] in spot_ids
    }
