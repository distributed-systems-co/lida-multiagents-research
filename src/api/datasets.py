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
        import os
        import fcntl
        import time
        clear_logs = os.getenv("CLEAR_LOGS_ON_START", "").lower() in ("true", "1", "yes")

        with self._transaction() as conn:
            # Clear LLM logs if requested (use lock + marker to ensure only first worker clears)
            if clear_logs:
                marker_file = Path("/tmp/.llm_logs_cleared")
                lock_file = Path("/tmp/.llm_logs_clear.lock")
                container_start = str(int(Path("/proc/1/stat").stat().st_mtime))

                try:
                    with open(lock_file, "w") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)  # Blocking lock
                        # Check marker while holding lock
                        should_clear = True
                        if marker_file.exists():
                            try:
                                if marker_file.read_text().strip() == container_start:
                                    should_clear = False
                            except Exception:
                                pass

                        if should_clear:
                            conn.execute("DROP TABLE IF EXISTS llm_response_log")
                            logger.info("Cleared LLM response logs (CLEAR_LOGS_ON_START=true)")
                            marker_file.write_text(container_start)
                except Exception as e:
                    logger.warning(f"Could not clear logs: {e}")

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

                CREATE TABLE IF NOT EXISTS llm_response_log (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    agent_name TEXT,
                    model_requested TEXT,
                    model_actual TEXT,
                    prompt TEXT,
                    response TEXT,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    timestamp TEXT NOT NULL,
                    duration_ms INTEGER,
                    deliberation_id TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_llm_agent ON llm_response_log(agent_id);
                CREATE INDEX IF NOT EXISTS idx_llm_model ON llm_response_log(model_actual);
                CREATE INDEX IF NOT EXISTS idx_llm_timestamp ON llm_response_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_llm_deliberation ON llm_response_log(deliberation_id);

                CREATE TABLE IF NOT EXISTS deliberations (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    phase TEXT,
                    current_round INTEGER DEFAULT 0,
                    max_rounds INTEGER DEFAULT 7,
                    scenario_name TEXT,
                    config_json TEXT,
                    consensus_json TEXT,
                    vote_history_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_deliberation_status ON deliberations(status);
                CREATE INDEX IF NOT EXISTS idx_deliberation_created ON deliberations(created_at);
            """)

            # Migration: Add vote_history_json column if it doesn't exist (for existing databases)
            try:
                conn.execute("ALTER TABLE deliberations ADD COLUMN vote_history_json TEXT")
                logger.info("Added vote_history_json column to deliberations table")
            except sqlite3.OperationalError:
                pass  # Column already exists

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

    # LLM response logging
    def log_llm_response(
        self,
        agent_id: Optional[str],
        model_requested: str,
        model_actual: str,
        prompt: str,
        response: str,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        duration_ms: Optional[int] = None,
        full_logs: bool = False,
        agent_name: Optional[str] = None,
        deliberation_id: Optional[str] = None,
    ):
        """Log an LLM response.

        Args:
            agent_id: ID of the agent making the request
            model_requested: Model that was requested
            model_actual: Model that actually responded
            prompt: The prompt sent (truncated to 200 chars unless full_logs=True)
            response: The response received (truncated to 200 chars unless full_logs=True)
            tokens_in: Input token count
            tokens_out: Output token count
            duration_ms: Request duration in milliseconds
            full_logs: If True, store complete content; if False, truncate to 200 chars
            agent_name: Human-readable name of the agent (e.g., "Elon Musk")
            deliberation_id: ID of the deliberation this log belongs to
        """
        log_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Truncate content unless full_logs is enabled
        if not full_logs:
            prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
            response = response[:200] + "..." if len(response) > 200 else response

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO llm_response_log
                (id, agent_id, agent_name, model_requested, model_actual, prompt, response,
                 tokens_in, tokens_out, timestamp, duration_ms, deliberation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (log_id, agent_id, agent_name, model_requested, model_actual, prompt, response,
                 tokens_in, tokens_out, now, duration_ms, deliberation_id),
            )

    def get_llm_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        agent_id: Optional[str] = None,
        model: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        deliberation_id: Optional[str] = None,
    ) -> list[dict]:
        """Get LLM response logs with optional filters.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            agent_id: Filter by agent ID
            model: Filter by model name (matches model_actual)
            start_time: Filter by timestamp >= start_time (ISO format)
            end_time: Filter by timestamp <= end_time (ISO format)
            deliberation_id: Filter by deliberation ID
        """
        query = "SELECT * FROM llm_response_log WHERE 1=1"
        params: list[Any] = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if model:
            query += " AND model_actual LIKE ?"
            params.append(f"%{model}%")
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        if deliberation_id:
            query += " AND deliberation_id = ?"
            params.append(deliberation_id)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._transaction() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": row["id"],
                "agent_id": row["agent_id"],
                "agent_name": row["agent_name"],
                "model_requested": row["model_requested"],
                "model_actual": row["model_actual"],
                "prompt": row["prompt"],
                "response": row["response"],
                "tokens_in": row["tokens_in"],
                "tokens_out": row["tokens_out"],
                "timestamp": row["timestamp"],
                "duration_ms": row["duration_ms"],
                "deliberation_id": row["deliberation_id"],
            }
            for row in rows
        ]

    def get_llm_log_count(
        self,
        agent_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> int:
        """Get count of LLM response logs."""
        query = "SELECT COUNT(*) as c FROM llm_response_log WHERE 1=1"
        params: list[Any] = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if model:
            query += " AND model_actual LIKE ?"
            params.append(f"%{model}%")

        with self._transaction() as conn:
            return conn.execute(query, params).fetchone()["c"]

    def get_stats(self) -> dict:
        """Get storage statistics."""
        with self._transaction() as conn:
            datasets = conn.execute("SELECT COUNT(*) as c FROM datasets").fetchone()["c"]
            records = conn.execute("SELECT COUNT(*) as c FROM records").fetchone()["c"]
            chats = conn.execute("SELECT COUNT(*) as c FROM chat_history").fetchone()["c"]
            messages = conn.execute("SELECT COUNT(*) as c FROM message_log").fetchone()["c"]
            llm_logs = conn.execute("SELECT COUNT(*) as c FROM llm_response_log").fetchone()["c"]

        return {
            "datasets": datasets,
            "total_records": records,
            "chat_messages": chats,
            "logged_messages": messages,
            "llm_response_logs": llm_logs,
        }

    # Deliberation operations
    def create_deliberation(
        self,
        topic: str,
        scenario_name: Optional[str] = None,
        config: Optional[dict] = None,
        max_rounds: int = 7,
    ) -> dict:
        """Create a new deliberation.

        Args:
            topic: The topic to deliberate
            scenario_name: Optional scenario name
            config: Optional configuration dict
            max_rounds: Maximum number of rounds (default 7)

        Returns:
            The created deliberation as a dict
        """
        deliberation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO deliberations (id, topic, status, phase, current_round, max_rounds,
                                          scenario_name, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    deliberation_id,
                    topic,
                    "pending",
                    "",
                    0,
                    max_rounds,
                    scenario_name,
                    json.dumps(config) if config else None,
                    now,
                ),
            )

        logger.info(f"Created deliberation: {deliberation_id} - {topic[:50]}...")
        return self.get_deliberation(deliberation_id)

    def get_deliberation(self, deliberation_id: str) -> Optional[dict]:
        """Get a deliberation by ID."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM deliberations WHERE id = ?", (deliberation_id,)
            ).fetchone()

        if not row:
            return None

        # Handle vote_history_json column which may not exist in older databases
        try:
            vote_history = json.loads(row["vote_history_json"]) if row["vote_history_json"] else []
        except (KeyError, TypeError):
            vote_history = []

        return {
            "id": row["id"],
            "topic": row["topic"],
            "status": row["status"],
            "phase": row["phase"],
            "current_round": row["current_round"],
            "max_rounds": row["max_rounds"],
            "scenario_name": row["scenario_name"],
            "config": json.loads(row["config_json"]) if row["config_json"] else None,
            "consensus": json.loads(row["consensus_json"]) if row["consensus_json"] else None,
            "vote_history": vote_history,
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }

    def list_deliberations(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List deliberations with optional status filter.

        Args:
            status: Filter by status (pending, active, paused, completed, stopped)
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of deliberation dicts
        """
        query = "SELECT * FROM deliberations WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._transaction() as conn:
            rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            # Handle vote_history_json column which may not exist in older databases
            try:
                vote_history = json.loads(row["vote_history_json"]) if row["vote_history_json"] else []
            except (KeyError, TypeError):
                vote_history = []

            results.append({
                "id": row["id"],
                "topic": row["topic"],
                "status": row["status"],
                "phase": row["phase"],
                "current_round": row["current_round"],
                "max_rounds": row["max_rounds"],
                "scenario_name": row["scenario_name"],
                "config": json.loads(row["config_json"]) if row["config_json"] else None,
                "consensus": json.loads(row["consensus_json"]) if row["consensus_json"] else None,
                "vote_history": vote_history,
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
            })
        return results

    def update_deliberation(
        self,
        deliberation_id: str,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        current_round: Optional[int] = None,
        consensus: Optional[dict] = None,
        vote_history: Optional[list] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> bool:
        """Update a deliberation's state.

        Args:
            deliberation_id: The deliberation ID
            status: New status (pending, active, paused, completed, stopped)
            phase: Current phase
            current_round: Current round number
            consensus: Consensus dict
            vote_history: List of vote history dicts per round
            started_at: When deliberation started (ISO format)
            completed_at: When deliberation completed (ISO format)

        Returns:
            True if updated, False if not found
        """
        updates = []
        params: list[Any] = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if phase is not None:
            updates.append("phase = ?")
            params.append(phase)
        if current_round is not None:
            updates.append("current_round = ?")
            params.append(current_round)
        if consensus is not None:
            updates.append("consensus_json = ?")
            params.append(json.dumps(consensus))
        if vote_history is not None:
            updates.append("vote_history_json = ?")
            params.append(json.dumps(vote_history))
        if started_at is not None:
            updates.append("started_at = ?")
            params.append(started_at)
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)

        if not updates:
            return False

        params.append(deliberation_id)
        query = f"UPDATE deliberations SET {', '.join(updates)} WHERE id = ?"

        with self._transaction() as conn:
            result = conn.execute(query, params)
            return result.rowcount > 0

    def delete_deliberation(self, deliberation_id: str) -> bool:
        """Delete a deliberation and its associated logs.

        Args:
            deliberation_id: The deliberation ID

        Returns:
            True if deleted, False if not found
        """
        with self._transaction() as conn:
            # Delete associated LLM logs
            conn.execute(
                "DELETE FROM llm_response_log WHERE deliberation_id = ?",
                (deliberation_id,)
            )
            # Delete the deliberation
            result = conn.execute(
                "DELETE FROM deliberations WHERE id = ?",
                (deliberation_id,)
            )
            return result.rowcount > 0
