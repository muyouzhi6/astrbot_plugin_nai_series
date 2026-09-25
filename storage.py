"""Atomic, credential-free session and preview storage."""

import hashlib
import json
import sqlite3
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
        self.database.chmod(0o600)

    def connect(self):
        return sqlite3.connect(self.database, timeout=10)

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
