"""Atomic, credential-free session and preview storage."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.previews = root / "previews"
        self.previews.mkdir(exist_ok=True)
        self.database = root / "state.sqlite3"
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, created REAL NOT NULL, value TEXT NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, created)")
        self.database.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO state VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def preview_path(self, preset):
        key = hashlib.sha256(f"{preset.family}:{preset.id}".encode()).hexdigest()
        return self.previews / f"{key}.png"

    def save_job(self, job):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?)",
                (job["id"], job["owner"], job["created"], json.dumps(job, ensure_ascii=False)),
            )

    def save_jobs(self, jobs):
        with self.connect() as db:
            db.executemany(
                "INSERT INTO jobs VALUES (?, ?, ?, ?)",
                [
                    (job["id"], job["owner"], job["created"], json.dumps(job, ensure_ascii=False))
                    for job in jobs
                ],
            )

    def claim(self, key):
        with self.connect() as db:
            result = db.execute("INSERT OR IGNORE INTO state VALUES (?, ?)", (key, "true"))
            return result.rowcount == 1

    def jobs(self, owner=None, limit=None, origin=None):
        query, args = "SELECT value FROM jobs", []
        if owner is not None:
            query += " WHERE owner=?"
            args.append(owner)
        if origin is not None:
            query += " AND " if owner is not None else " WHERE "
            query += "json_extract(value, '$.origin')=?"
            args.append(origin)
        query += " ORDER BY created DESC"
        if limit is not None:
            query += " LIMIT ?"
            args.append(limit)
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(query, args)]

    def job(self, task_id, owner):
        with self.connect() as db:
            row = db.execute(
                "SELECT value FROM jobs WHERE id=? AND owner=?", (task_id, owner)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def image_path(self, task_id):
        folder = self.root / "outputs"
        folder.mkdir(exist_ok=True)
        return folder / (hashlib.sha256(task_id.encode()).hexdigest() + ".png")

    def prune_jobs(self, cutoff, active):
        for job in self.jobs():
            if job["state"] in active or job.get("updated", job["created"]) >= cutoff:
                continue
            self.image_path(job["id"]).unlink(missing_ok=True)
            self.image_path(job["id"]).with_suffix(".tmp").unlink(missing_ok=True)
            with self.connect() as db:
                db.execute("DELETE FROM jobs WHERE id=?", (job["id"],))
                db.execute(
                    "DELETE FROM state WHERE key=?",
                    ("completion:" + (job.get("batch_id") or job["id"]),),
                )
