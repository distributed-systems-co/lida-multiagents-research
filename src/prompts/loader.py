"""Prompt loader for LIDA multi-agent system.

Redesigned for thousands of prompts with:
- Auto-discovery of prompt files
- Lazy loading with file indexing
- SQLite backend for fast queries
- Memory-efficient iteration
"""
from __future__ import annotations

import re
import logging
import sqlite3
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Iterator, List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class Prompt:
    """A system prompt with metadata."""
    id: int
    text: str
    category: str
    subcategory: str = ""
    tags: List[str] = field(default_factory=list)
    file_path: str = ""

    def __repr__(self):
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Prompt({self.id}, {self.category}/{self.subcategory}: {preview})"


class PromptIndex:
    """SQLite-backed index for fast prompt queries."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT DEFAULT '',
                    tags TEXT DEFAULT '',
                    file_path TEXT DEFAULT '',
                    text_hash TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_category ON prompts(category);
                CREATE INDEX IF NOT EXISTS idx_subcategory ON prompts(subcategory);
                CREATE VIRTUAL TABLE IF NOT EXISTS prompts_fts USING fts5(
                    text, category, subcategory, tags,
                    content='prompts',
                    content_rowid='id'
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_file_hash(self, filepath: Path) -> Optional[str]:
        """Get stored hash for a file."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                (f"hash:{filepath}",)
            ).fetchone()
            return row[0] if row else None

    def set_file_hash(self, filepath: Path, hash_val: str):
        """Store hash for a file."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (f"hash:{filepath}", hash_val)
            )

    def clear_category(self, category: str):
        """Remove all prompts from a category."""
        with self._conn() as conn:
            conn.execute("DELETE FROM prompts WHERE category = ?", (category,))
            conn.execute("DELETE FROM prompts_fts WHERE category = ?", (category,))

    def insert_prompt(self, prompt: Prompt):
        """Insert a prompt into the index."""
        text_hash = hashlib.md5(prompt.text.encode()).hexdigest()[:16]
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO prompts
                (id, text, category, subcategory, tags, file_path, text_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt.id, prompt.text, prompt.category,
                prompt.subcategory, ",".join(prompt.tags),
                prompt.file_path, text_hash
            ))
            # Update FTS
            conn.execute("""
                INSERT INTO prompts_fts (rowid, text, category, subcategory, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (
                prompt.id, prompt.text, prompt.category,
                prompt.subcategory, ",".join(prompt.tags)
            ))

    def insert_batch(self, prompts: List[Prompt]):
        """Insert multiple prompts efficiently."""
        with self._conn() as conn:
            for prompt in prompts:
                text_hash = hashlib.md5(prompt.text.encode()).hexdigest()[:16]
                conn.execute("""
                    INSERT OR REPLACE INTO prompts
                    (id, text, category, subcategory, tags, file_path, text_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    prompt.id, prompt.text, prompt.category,
                    prompt.subcategory, ",".join(prompt.tags),
                    prompt.file_path, text_hash
                ))

    def rebuild_fts(self):
        """Rebuild the full-text search index."""
        with self._conn() as conn:
            conn.execute("DROP TABLE IF EXISTS prompts_fts")
            conn.execute("""
                CREATE VIRTUAL TABLE prompts_fts USING fts5(
                    text, category, subcategory, tags,
                    content='prompts',
                    content_rowid='id'
                )
            """)
            conn.execute("""
                INSERT INTO prompts_fts (rowid, text, category, subcategory, tags)
                SELECT id, text, category, subcategory, tags FROM prompts
            """)

    def get(self, prompt_id: int) -> Optional[Prompt]:
        """Get a prompt by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM prompts WHERE id = ?", (prompt_id,)
            ).fetchone()
            return self._row_to_prompt(row) if row else None

    def get_by_category(self, category: str) -> List[Prompt]:
        """Get all prompts in a category."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM prompts WHERE category = ? ORDER BY id",
                (category,)
            ).fetchall()
            return [self._row_to_prompt(r) for r in rows]

    def get_by_subcategory(self, subcategory: str) -> List[Prompt]:
        """Get all prompts in a subcategory."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM prompts WHERE subcategory = ? ORDER BY id",
                (subcategory,)
            ).fetchall()
            return [self._row_to_prompt(r) for r in rows]

    def search(self, query: str, limit: int = 50) -> List[Prompt]:
        """Full-text search prompts."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT p.* FROM prompts p
                JOIN prompts_fts fts ON p.id = fts.rowid
                WHERE prompts_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            return [self._row_to_prompt(r) for r in rows]

    def random(self, category: Optional[str] = None, n: int = 1) -> List[Prompt]:
        """Get random prompts."""
        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM prompts WHERE category = ? ORDER BY RANDOM() LIMIT ?",
                    (category, n)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM prompts ORDER BY RANDOM() LIMIT ?", (n,)
                ).fetchall()
            return [self._row_to_prompt(r) for r in rows]

    def categories(self) -> Dict[str, int]:
        """Get category names and counts."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM prompts GROUP BY category ORDER BY category"
            ).fetchall()
            return {r['category']: r['cnt'] for r in rows}

    def subcategories(self, category: Optional[str] = None) -> Dict[str, int]:
        """Get subcategory names and counts."""
        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    "SELECT subcategory, COUNT(*) as cnt FROM prompts WHERE category = ? GROUP BY subcategory",
                    (category,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT subcategory, COUNT(*) as cnt FROM prompts GROUP BY subcategory"
                ).fetchall()
            return {r['subcategory']: r['cnt'] for r in rows if r['subcategory']}

    def count(self, category: Optional[str] = None) -> int:
        """Get total prompt count."""
        with self._conn() as conn:
            if category:
                row = conn.execute(
                    "SELECT COUNT(*) FROM prompts WHERE category = ?", (category,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()
            return row[0]

    def iterate(self, category: Optional[str] = None, batch_size: int = 100) -> Iterator[Prompt]:
        """Memory-efficient iteration over prompts."""
        with self._conn() as conn:
            if category:
                cursor = conn.execute(
                    "SELECT * FROM prompts WHERE category = ? ORDER BY id",
                    (category,)
                )
            else:
                cursor = conn.execute("SELECT * FROM prompts ORDER BY id")

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield self._row_to_prompt(row)

    def _row_to_prompt(self, row: sqlite3.Row) -> Prompt:
        return Prompt(
            id=row['id'],
            text=row['text'],
            category=row['category'],
            subcategory=row['subcategory'] or "",
            tags=row['tags'].split(",") if row['tags'] else [],
            file_path=row['file_path'] or ""
        )


class PromptLoader:
    """Loads and manages system prompts with auto-discovery and lazy loading."""

    # Pattern to extract category from filename: prompts_NN_category_name.txt
    FILENAME_PATTERN = re.compile(r'prompts_(?:\d+_)?(.+)\.txt$')

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        index_path: Optional[Path] = None,
        auto_load: bool = True
    ):
        """Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing prompt files.
            index_path: Path for SQLite index. Defaults to prompts_dir/prompts.db
            auto_load: Whether to auto-load on first access.
        """
        # Find prompts directory
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        else:
            # Try multiple locations
            candidates = [
                Path.cwd() / "populations.prompts",
                Path(__file__).parent.parent.parent / "populations.prompts",
                Path("/Users/arthurcolle/prompts_context/populations.prompts"),
            ]
            self.prompts_dir = next((p for p in candidates if p.exists()), candidates[0])

        # Index path
        if index_path:
            self.index_path = Path(index_path)
        else:
            self.index_path = self.prompts_dir.parent / "data" / "prompts.db"

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._index = PromptIndex(self.index_path)
        self._demiurge_prompt: Optional[str] = None
        self._loaded = False
        self._auto_load = auto_load

        # Discovered files
        self._prompt_files: List[Path] = []

    def discover_files(self) -> List[Path]:
        """Discover all prompt files."""
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return []

        self._prompt_files = sorted(self.prompts_dir.glob("prompts_*.txt"))
        logger.info(f"Discovered {len(self._prompt_files)} prompt files in {self.prompts_dir}")
        return self._prompt_files

    def _category_from_filename(self, filepath: Path) -> str:
        """Extract category name from filename."""
        match = self.FILENAME_PATTERN.search(filepath.name)
        if match:
            return match.group(1).replace("_", " ").title().replace(" ", "_").lower()
        return filepath.stem

    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute hash of file for change detection."""
        stat = filepath.stat()
        return hashlib.md5(f"{filepath}:{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()

    def load(self, force: bool = False) -> int:
        """Load all prompts, updating index only for changed files.

        Args:
            force: Force reload all files regardless of cache.

        Returns:
            Number of prompts loaded/indexed.
        """
        if self._loaded and not force:
            return self._index.count()

        self.discover_files()
        total = 0
        files_updated = 0

        # Load demiurge baseline
        demiurge_candidates = [
            self.prompts_dir.parent / "demiurge.agent.baseline.md",
            Path.cwd() / "demiurge.agent.baseline.md",
        ]
        for demiurge_path in demiurge_candidates:
            if demiurge_path.exists():
                self._demiurge_prompt = demiurge_path.read_text()
                logger.info(f"Loaded Demiurge baseline ({len(self._demiurge_prompt)} chars)")
                break

        # Process each prompt file
        for filepath in self._prompt_files:
            file_hash = self._compute_file_hash(filepath)
            stored_hash = self._index.get_file_hash(filepath)

            if not force and stored_hash == file_hash:
                # File unchanged, skip
                continue

            category = self._category_from_filename(filepath)

            # Clear old prompts for this category
            self._index.clear_category(category)

            # Parse and index
            prompts = list(self._parse_file(filepath, category))
            if prompts:
                self._index.insert_batch(prompts)
                self._index.set_file_hash(filepath, file_hash)
                logger.info(f"Indexed {len(prompts)} prompts from {filepath.name} -> {category}")
                files_updated += 1

            total += len(prompts)

        self._loaded = True

        # Rebuild FTS if any files were updated
        if files_updated > 0:
            self._index.rebuild_fts()

        total_count = self._index.count()

        if files_updated > 0:
            logger.info(f"Updated {files_updated} files, total {total_count} prompts indexed")
        else:
            logger.info(f"Index up-to-date: {total_count} prompts")

        return total_count

    def _parse_file(self, filepath: Path, category: str) -> Iterator[Prompt]:
        """Parse a prompt file, yielding prompts."""
        content = filepath.read_text()
        current_subcategory = ""

        # Patterns
        subcategory_pattern = re.compile(r'^##\s+(.+?)\s*\(\d+\s*prompts?\)', re.IGNORECASE)
        prompt_pattern = re.compile(r'^(\d+)\.\s+(.+)', re.DOTALL)

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Subcategory header
            sub_match = subcategory_pattern.match(line)
            if sub_match:
                current_subcategory = sub_match.group(1).strip()
                i += 1
                continue

            # Numbered prompt
            prompt_match = prompt_pattern.match(line)
            if prompt_match:
                prompt_id = int(prompt_match.group(1))
                prompt_text = prompt_match.group(2).strip()

                # Continue reading multi-line prompt
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line:
                        i += 1
                        continue
                    if prompt_pattern.match(next_line) or subcategory_pattern.match(next_line):
                        break
                    if next_line.startswith('#'):
                        break
                    prompt_text += " " + next_line
                    i += 1

                yield Prompt(
                    id=prompt_id,
                    text=prompt_text.strip(),
                    category=category,
                    subcategory=current_subcategory,
                    tags=self._extract_tags(prompt_text),
                    file_path=str(filepath)
                )
            else:
                i += 1

    def _extract_tags(self, text: str) -> List[str]:
        """Extract role/expertise tags from prompt text."""
        tags = []
        text_lower = text.lower()

        role_keywords = [
            "specialist", "researcher", "historian", "theorist", "analyst",
            "practitioner", "expert", "philosopher", "scientist", "engineer",
            "designer", "artist", "critic", "educator", "teacher", "developer",
            "architect", "manager", "consultant", "strategist", "operator"
        ]

        for keyword in role_keywords:
            if keyword in text_lower:
                tags.append(keyword)

        return tags[:5]  # Limit tags

    def _ensure_loaded(self):
        """Ensure prompts are loaded."""
        if not self._loaded and self._auto_load:
            self.load()

    @property
    def demiurge_prompt(self) -> str:
        """Get the Demiurge baseline system prompt."""
        self._ensure_loaded()
        return self._demiurge_prompt or "You are the Demiurge, a craftsman-intelligence."

    def get(self, prompt_id: int) -> Optional[Prompt]:
        """Get a prompt by ID."""
        self._ensure_loaded()
        return self._index.get(prompt_id)

    def get_by_category(self, category: str) -> List[Prompt]:
        """Get all prompts in a category."""
        self._ensure_loaded()
        return self._index.get_by_category(category)

    def get_by_subcategory(self, subcategory: str) -> List[Prompt]:
        """Get all prompts in a subcategory."""
        self._ensure_loaded()
        return self._index.get_by_subcategory(subcategory)

    def search(self, query: str, limit: int = 50) -> List[Prompt]:
        """Full-text search prompts."""
        self._ensure_loaded()
        return self._index.search(query, limit)

    def random(self, category: Optional[str] = None, n: int = 1) -> List[Prompt]:
        """Get random prompt(s)."""
        self._ensure_loaded()
        return self._index.random(category, n)

    def categories(self) -> Dict[str, int]:
        """Get category names and prompt counts."""
        self._ensure_loaded()
        return self._index.categories()

    def subcategories(self, category: Optional[str] = None) -> Dict[str, int]:
        """Get subcategory names and prompt counts."""
        self._ensure_loaded()
        return self._index.subcategories(category)

    def count(self, category: Optional[str] = None) -> int:
        """Get total prompt count."""
        self._ensure_loaded()
        return self._index.count(category)

    def iterate(self, category: Optional[str] = None) -> Iterator[Prompt]:
        """Memory-efficient iteration over prompts."""
        self._ensure_loaded()
        return self._index.iterate(category)

    def all_prompts(self) -> List[Prompt]:
        """Get all prompts (use iterate() for large datasets)."""
        self._ensure_loaded()
        return list(self._index.iterate())

    def stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        self._ensure_loaded()
        cats = self.categories()
        return {
            "total_prompts": self._index.count(),
            "categories": len(cats),
            "category_counts": cats,
            "prompt_files": len(self._prompt_files),
            "index_path": str(self.index_path),
            "prompts_dir": str(self.prompts_dir),
        }


# Global loader instance
_loader: Optional[PromptLoader] = None


def get_loader() -> PromptLoader:
    """Get the global prompt loader instance."""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader


# Legacy compatibility
class PromptCategory:
    """Dynamic category accessor for backwards compatibility."""

    def __init__(self):
        self._categories: Dict[str, str] = {}

    def __getattr__(self, name: str) -> str:
        loader = get_loader()
        cats = loader.categories()
        # Try exact match
        if name.lower() in cats:
            return name.lower()
        # Try with underscores
        name_lower = name.lower().replace("_", " ")
        for cat in cats:
            if cat.replace("_", " ") == name_lower:
                return cat
        raise AttributeError(f"Category '{name}' not found. Available: {list(cats.keys())}")
