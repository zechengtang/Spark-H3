"""SQLite-backed storage for the many small JSON records made by experiments.

Large artifacts (latents, videos, traces) should remain ordinary files.  This
module is for the small, numerous records that would otherwise consume one
shared-filesystem inode each.  A store uses one database plus SQLite's two WAL
sidecars while writers are active, and supports concurrent worker processes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS json_records (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_ns INTEGER NOT NULL
) WITHOUT ROWID
"""


def _normalize_key(key: str | os.PathLike[str]) -> str:
    value = Path(key).as_posix()
    if not value or value == "." or value.startswith("/"):
        raise ValueError(f"record key must be a non-empty relative path: {key!r}")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"record key must not contain empty, '.' or '..' parts: {key!r}")
    return value


class JsonRecordStore:
    """A process-safe key/JSON store backed by a single SQLite database.

    Connections are lazy and process-local, so an instance may safely exist
    before a process is forked.  SQLite WAL mode allows independent experiment
    workers to commit records concurrently without temporary JSON files.
    """

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 120.0):
        self.path = Path(path)
        self.timeout = timeout
        self._local = threading.local()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._new_connection()
        try:
            connection.execute(_SCHEMA)
        finally:
            connection.close()

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout,
            isolation_level=None,
        )
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        # Setting journal_mode needs an exclusive lock.  Workers can all open
        # the store at nearly the same instant, and this PRAGMA does not honor
        # busy_timeout consistently across SQLite builds, so retry explicitly.
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                if mode.lower() != "wal":
                    connection.execute("PRAGMA journal_mode = WAL")
                break
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower() or time.monotonic() >= deadline:
                    connection.close()
                    raise
                time.sleep(0.05)
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA wal_autocheckpoint = 1000")
        return connection

    def _connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        connection = getattr(self._local, "connection", None)
        if connection is None or getattr(self._local, "pid", None) != pid:
            if connection is not None:
                connection.close()
            connection = self._new_connection()
            self._local.connection = connection
            self._local.pid = pid
        return connection

    def put(self, key: str | os.PathLike[str], payload: Any) -> None:
        normalized = _normalize_key(key)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=True,
            separators=(",", ":"),
        )
        self._connection().execute(
            """
            INSERT INTO json_records(key, payload, updated_ns)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                payload = excluded.payload,
                updated_ns = excluded.updated_ns
            """,
            (normalized, encoded, time.time_ns()),
        )

    def put_many(self, records: Iterable[tuple[str, Any]]) -> None:
        rows = [
            (
                _normalize_key(key),
                json.dumps(value, ensure_ascii=False, allow_nan=True, separators=(",", ":")),
                time.time_ns(),
            )
            for key, value in records
        ]
        if not rows:
            return
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.executemany(
                """
                INSERT INTO json_records(key, payload, updated_ns)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    payload = excluded.payload,
                    updated_ns = excluded.updated_ns
                """,
                rows,
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def get(self, key: str | os.PathLike[str]) -> Any:
        normalized = _normalize_key(key)
        row = self._connection().execute(
            "SELECT payload FROM json_records WHERE key = ?", (normalized,)
        ).fetchone()
        if row is None:
            raise KeyError(normalized)
        return json.loads(row[0])

    def contains(self, key: str | os.PathLike[str]) -> bool:
        normalized = _normalize_key(key)
        return self._connection().execute(
            "SELECT 1 FROM json_records WHERE key = ?", (normalized,)
        ).fetchone() is not None

    def delete(self, key: str | os.PathLike[str]) -> bool:
        normalized = _normalize_key(key)
        cursor = self._connection().execute(
            "DELETE FROM json_records WHERE key = ?", (normalized,)
        )
        return cursor.rowcount > 0

    def items(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        if prefix:
            prefix = _normalize_key(prefix).rstrip("/") + "/"
            rows = self._connection().execute(
                """
                SELECT key, payload FROM json_records
                WHERE key >= ? AND key < ? ORDER BY key
                """,
                (prefix, prefix + "\U0010ffff"),
            ).fetchall()
        else:
            rows = self._connection().execute(
                "SELECT key, payload FROM json_records ORDER BY key"
            ).fetchall()
        for key, payload in rows:
            yield key, json.loads(payload)

    def count(self, prefix: str = "") -> int:
        if not prefix:
            return int(
                self._connection().execute(
                    "SELECT COUNT(*) FROM json_records"
                ).fetchone()[0]
            )
        prefix = _normalize_key(prefix).rstrip("/") + "/"
        return int(
            self._connection().execute(
                "SELECT COUNT(*) FROM json_records WHERE key >= ? AND key < ?",
                (prefix, prefix + "\U0010ffff"),
            ).fetchone()[0]
        )

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def checkpoint(self) -> None:
        """Flush committed WAL pages into the database after workers exit."""
        self._connection().execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def __enter__(self) -> "JsonRecordStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def generation_key(arm: str, case: int) -> str:
    return f"generation/{arm}/{case:02d}"


def quality_key(arm: str, case: int) -> str:
    return f"quality/{arm}/{case:02d}"
