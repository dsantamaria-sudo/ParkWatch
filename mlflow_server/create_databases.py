"""ParkWatch — creates the MLflow role and databases on the PostgreSQL server (idempotent).

Role `mlflow` owns `parkwatch_mlflow` (tracking) and `parkwatch_mlflow_auth` (basic-auth users).
Other databases on the server (labelstudio) are not touched. Re-running it only resets the
password of the role.

Usage (passwords from the Key Vault in environment variables, firewall open for your IP):
  $env:PG_ADMIN_PASSWORD = ...; $env:PG_MLFLOW_PASSWORD = ...
  uv run --no-project --with "psycopg[binary]" python mlflow_server/create_databases.py
"""
from __future__ import annotations

import argparse
import os

import psycopg
from psycopg import sql


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="mlflowtest.postgres.database.azure.com")
    p.add_argument("--admin-user", default="mlflowtest")
    p.add_argument("--role", default="mlflow")
    p.add_argument("--databases", nargs="+", default=["parkwatch_mlflow", "parkwatch_mlflow_auth"])
    args = p.parse_args()

    role = sql.Identifier(args.role)
    password = sql.Literal(os.environ["PG_MLFLOW_PASSWORD"])
    with psycopg.connect(host=args.host, user=args.admin_user, password=os.environ["PG_ADMIN_PASSWORD"],
                         dbname="postgres", sslmode="require", autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (args.role,)).fetchone():
            conn.execute(sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(role, password))
            print(f"role {args.role}: password updated")
        else:
            conn.execute(sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD {}").format(role, password))
            print(f"role {args.role}: created")
        # The admin is not a superuser on Azure: it must be a member of the role to make it owner.
        conn.execute(sql.SQL("GRANT {} TO {}").format(role, sql.Identifier(args.admin_user)))

        for db in args.databases:
            if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,)).fetchone():
                print(f"database {db}: already exists")
            else:
                conn.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(db), role))
                print(f"database {db}: created")
            conn.execute(sql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(sql.Identifier(db)))

    # On Azure the `public` schema of a new database is not owned by the database owner, so the
    # role could not create tables (MLflow migrations fail with "permission denied for schema public").
    for db in args.databases:
        with psycopg.connect(host=args.host, user=args.admin_user, password=os.environ["PG_ADMIN_PASSWORD"],
                             dbname=db, sslmode="require", autocommit=True) as conn:
            conn.execute(sql.SQL("ALTER SCHEMA public OWNER TO {}").format(role))
            print(f"database {db}: schema public owned by {args.role}")

    with psycopg.connect(host=args.host, user=args.role, password=os.environ["PG_MLFLOW_PASSWORD"],
                         dbname=args.databases[0], sslmode="require") as conn:
        print("login OK as", conn.execute("SELECT current_user").fetchone()[0])


if __name__ == "__main__":
    main()
