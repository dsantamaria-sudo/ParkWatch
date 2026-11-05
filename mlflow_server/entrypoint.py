"""ParkWatch — starts the MLflow server from environment variables.

Secrets arrive as Container App secrets (Key Vault references), never baked into the image.

  PG_HOST, PG_USER, PG_PASSWORD     PostgreSQL server and the MLflow role
  PG_DB / PG_AUTH_DB                tracking database / basic-auth users database
                                    (separate: both use their own alembic_version table)
  ARTIFACTS_DESTINATION             wasbs://<container>@<account>.blob.core.windows.net/
  ALLOWED_HOSTS                     comma-separated Host headers accepted by the server
                                    (the public FQDN of the Container App)
  CORS_ALLOWED_ORIGINS              origins allowed to POST from a browser (https://<FQDN>):
                                    without it the UI's own search requests get 403
  MLFLOW_AUTH_ADMIN_PASSWORD        password of the `admin` user, only used on first start
  MLFLOW_FLASK_SERVER_SECRET_KEY    CSRF key of the basic-auth forms
  AZURE_CLIENT_ID                   user-assigned identity used to write artifacts
  MLFLOW_WORKERS                    uvicorn workers (default 2)
  MLFLOW_DB_UPGRADE                 "true" migrates the tracking schema before starting; only
                                    needed after an MLflow upgrade (adds ~10 s to every start)

MLflow reads the last three variables itself.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


def pg_uri(database: str) -> str:
    user, password = quote(os.environ["PG_USER"], safe=""), quote(os.environ["PG_PASSWORD"], safe="")
    return f"postgresql+psycopg://{user}:{password}@{os.environ['PG_HOST']}:5432/{database}?sslmode=require"


def main() -> None:
    auth_ini = Path("basic_auth.ini")
    auth_ini.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        f"database_uri = {pg_uri(os.environ['PG_AUTH_DB']).replace('%', '%%')}\n"
        "admin_username = admin\n"
        "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n"
        # Every request re-hashes the password (PBKDF2) unless cached: with the cache off a UI
        # page, which fires dozens of requests, takes seconds. A changed password may keep
        # working for up to this TTL.
        "auth_cache_ttl_seconds = 300\n",
        encoding="utf-8",
    )
    auth_ini.chmod(0o600)
    os.environ["MLFLOW_AUTH_CONFIG_PATH"] = str(auth_ini.resolve())

    # The server refuses a tracking database with an older schema (after an MLflow upgrade):
    # deploy once with MLFLOW_DB_UPGRADE=true. Not on every start: it imports MLflow one more
    # time and slows down every cold start of the scale-to-zero app.
    backend = pg_uri(os.environ["PG_DB"])
    if os.getenv("MLFLOW_DB_UPGRADE", "").lower() == "true":
        # No check=True: CalledProcessError would print the command line, password included.
        if subprocess.run(["mlflow", "db", "upgrade", backend]).returncode != 0:
            sys.exit("mlflow db upgrade failed (see the traceback above)")

    args = [
        "mlflow", "server",
        "--host", "0.0.0.0", "--port", "5000",
        "--workers", os.getenv("MLFLOW_WORKERS", "2"),
        "--backend-store-uri", backend,
        "--artifacts-destination", os.environ["ARTIFACTS_DESTINATION"],
        "--serve-artifacts",
        "--allowed-hosts", os.environ["ALLOWED_HOSTS"],
        "--cors-allowed-origins", os.environ["CORS_ALLOWED_ORIGINS"],
        "--app-name", "basic-auth",
    ]
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
