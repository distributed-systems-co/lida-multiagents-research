"""Dataset storage and management."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatasetStore:
    """SQLite-based dataset storage with JSON support."""

    def __init__(self, db_path: str = "data/datasets.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _transaction(self):
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._transaction() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    schema_def TEXT,
                    tags TEXT,
                    record_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_records_dataset ON records(dataset_id);

                CREATE TABLE IF NOT EXISTS chat_history (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chat_agent ON chat_history(agent_id);

                CREATE TABLE IF NOT EXISTS message_log (
                    id TEXT PRIMARY KEY,
                    msg_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    msg_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_msg_sender ON message_log(sender_id);
                CREATE INDEX IF NOT EXISTS idx_msg_recipient ON message_log(recipient_id);
            """)
        logger.info(f"Database initialized at {self.db_path}")

    # Dataset operations
    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        schema_def: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        """Create a new dataset."""
        dataset_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO datasets (id, name, description, schema_def, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    name,
                    description,
                    json.dumps(schema_def) if schema_def else None,
                    json.dumps(tags or []),
                    now,
                    now,
                ),
            )

        logger.info(f"Created dataset: {name} ({dataset_id})")
        return self.get_dataset(dataset_id)

    def get_dataset(self, dataset_id: str) -> Optional[dict]:
        """Get a dataset by ID."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
            ).fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "schema_def": json.loads(row["schema_def"]) if row["schema_def"] else None,
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "record_count": row["record_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_datasets(self) -> list[dict]:
        """List all datasets."""
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM datasets ORDER BY created_at DESC"
            ).fetchall()

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "schema_def": json.loads(row["schema_def"]) if row["schema_def"] else None,
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "record_count": row["record_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and all its records."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM records WHERE dataset_id = ?", (dataset_id,))
            result = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            return result.rowcount > 0

    # Record operations
    def add_records(self, dataset_id: str, records: list[dict]) -> int:
        """Add records to a dataset."""
        now = datetime.now(timezone.utc).isoformat()
        added = 0

        with self._transaction() as conn:
            for record in records:
                record_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO records (id, dataset_id, data, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (record_id, dataset_id, json.dumps(record), now),
                )
                added += 1

            conn.execute(
                """
                UPDATE datasets
                SET record_count = record_count + ?, updated_at = ?
                WHERE id = ?
                """,
                (added, now, dataset_id),
            )

        logger.info(f"Added {added} records to dataset {dataset_id}")
        return added

    def get_records(
        self,
        dataset_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get records from a dataset."""
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM records
                WHERE dataset_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (dataset_id, limit, offset),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "data": json.loads(row["data"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # Chat history
    def save_chat_message(self, agent_id: str, role: str, content: str):
        """Save a chat message."""
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO chat_history (id, agent_id, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (msg_id, agent_id, role, content, now),
            )

    def get_chat_history(
        self,
        agent_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get chat history for an agent."""
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chat_history
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (agent_id, limit),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "agent_id": row["agent_id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            }
            for row in reversed(rows)  # Return in chronological order
        ]

    # Message logging
    def log_message(
        self,
        msg_id: str,
        sender_id: str,
        recipient_id: str,
        msg_type: str,
        payload: dict,
    ):
        """Log a message."""
        log_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO message_log (id, msg_id, sender_id, recipient_id, msg_type, payload, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (log_id, msg_id, sender_id, recipient_id, msg_type, json.dumps(payload), now),
            )

    def get_message_log(
        self,
        limit: int = 100,
        sender_id: Optional[str] = None,
        recipient_id: Optional[str] = None,
    ) -> list[dict]:
        """Get message log with optional filters."""
        query = "SELECT * FROM message_log WHERE 1=1"
        params = []

        if sender_id:
            query += " AND sender_id = ?"
            params.append(sender_id)
        if recipient_id:
            query += " AND recipient_id = ?"
            params.append(recipient_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._transaction() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": row["id"],
                "msg_id": row["msg_id"],
                "sender_id": row["sender_id"],
                "recipient_id": row["recipient_id"],
                "msg_type": row["msg_type"],
                "payload": json.loads(row["payload"]),
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    def get_stats(self) -> dict:
        """Get storage statistics."""
        with self._transaction() as conn:
            datasets = conn.execute("SELECT COUNT(*) as c FROM datasets").fetchone()["c"]
            records = conn.execute("SELECT COUNT(*) as c FROM records").fetchone()["c"]
            chats = conn.execute("SELECT COUNT(*) as c FROM chat_history").fetchone()["c"]
            messages = conn.execute("SELECT COUNT(*) as c FROM message_log").fetchone()["c"]

        return {
            "datasets": datasets,
            "total_records": records,
            "chat_messages": chats,
            "logged_messages": messages,
        }
