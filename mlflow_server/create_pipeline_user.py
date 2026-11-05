"""ParkWatch — creates the MLflow user the pipeline writes with (idempotent).

As `admin`: creates (or resets the password of) the `pipeline` user, creates the training
experiment if missing and grants `pipeline` EDIT on it. Every other user keeps the server's
default (READ). Finally logs in as `pipeline` and creates a throw-away run with a metric in the
experiment, to check the permission end to end (admin deletes it afterwards).

Usage (passwords from the Key Vault in environment variables):
  $env:MLFLOW_ADMIN_PASSWORD = ...; $env:MLFLOW_PIPELINE_PASSWORD = ...
  uv run --no-project --with requests python mlflow_server/create_pipeline_user.py --url https://<app>
"""
from __future__ import annotations

import argparse
import os
import time

import requests


class Api:
    def __init__(self, url: str, user: str, password: str):
        self.url, self.session = url.rstrip("/"), requests.Session()
        self.session.auth = (user, password)

    def call(self, method: str, path: str, ok: tuple[int, ...] = (200,), **kwargs) -> requests.Response:
        response = self.session.request(method, f"{self.url}{path}", timeout=120, **kwargs)
        if response.status_code not in ok:
            raise SystemExit(f"{method} {path}: {response.status_code} {response.text[:300]}")
        return response


def wait_until_up(url: str) -> None:
    """The app scales to zero: the first request may arrive before the server listens."""
    for _ in range(30):
        try:
            if requests.get(f"{url}/health", timeout=30).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise SystemExit(f"{url} does not answer /health")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="MLflow server, https://<FQDN of the Container App>")
    p.add_argument("--user", default="pipeline")
    p.add_argument("--experiment", default="parkwatch-training")
    args = p.parse_args()

    wait_until_up(args.url.rstrip("/"))
    admin = Api(args.url, "admin", os.environ["MLFLOW_ADMIN_PASSWORD"])
    password = os.environ["MLFLOW_PIPELINE_PASSWORD"]

    r = admin.call("POST", "/api/2.0/mlflow/users/create", ok=(200, 400),
                   json={"username": args.user, "password": password})
    if r.status_code == 200:
        print(f"user {args.user}: created")
    elif "RESOURCE_ALREADY_EXISTS" in r.text:
        admin.call("PATCH", "/api/2.0/mlflow/users/update-password", json={"username": args.user, "password": password})
        print(f"user {args.user}: password updated")
    else:
        raise SystemExit(f"create user: {r.text[:300]}")

    r = admin.call("GET", "/api/2.0/mlflow/experiments/get-by-name", ok=(200, 404),
                   params={"experiment_name": args.experiment})
    if r.status_code == 200:
        experiment_id = r.json()["experiment"]["experiment_id"]
        print(f"experiment {args.experiment}: exists (id {experiment_id})")
    else:
        experiment_id = admin.call("POST", "/api/2.0/mlflow/experiments/create",
                                   json={"name": args.experiment}).json()["experiment_id"]
        print(f"experiment {args.experiment}: created (id {experiment_id})")

    # MLflow 3.x permission API: resource type + id instead of experiment-permissions/*.
    r = admin.call("POST", "/api/3.0/mlflow/users/permissions/grant", ok=(200, 400), json={
        "username": args.user, "resource_type": "experiment", "resource_id": experiment_id, "permission": "EDIT"})
    if r.status_code == 200:
        print(f"user {args.user}: EDIT on {args.experiment}")
    elif "RESOURCE_ALREADY_EXISTS" in r.text:
        print(f"user {args.user}: already had a permission on {args.experiment} (checked below)")
    else:
        raise SystemExit(f"grant permission: {r.text[:300]}")

    pipeline = Api(args.url, args.user, password)
    run = pipeline.call("POST", "/api/2.0/mlflow/runs/create", json={
        "experiment_id": experiment_id, "run_name": "permission-check", "start_time": int(time.time() * 1000)}).json()
    run_id = run["run"]["info"]["run_id"]
    pipeline.call("POST", "/api/2.0/mlflow/runs/log-metric", json={
        "run_id": run_id, "key": "check", "value": 1, "timestamp": int(time.time() * 1000), "step": 0})
    print(f"login OK as {args.user}: run created and metric logged in {args.experiment}")

    # EDIT does not include deleting runs (MANAGE does), and the pipeline never deletes:
    # admin removes this check run and any left by previous executions.
    checks = admin.call("POST", "/api/2.0/mlflow/runs/search", json={
        "experiment_ids": [experiment_id], "filter": "attributes.run_name = 'permission-check'"}).json()
    for r in checks.get("runs", []):
        admin.call("POST", "/api/2.0/mlflow/runs/delete", json={"run_id": r["info"]["run_id"]})
    print(f"check runs deleted: {len(checks.get('runs', []))}")


if __name__ == "__main__":
    main()
