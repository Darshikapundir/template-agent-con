"""Async Postgres repository for user rules.

Uses ``psycopg`` (async) against the same database that stores
LangGraph checkpoints. Tables are created lazily on first use via
:meth:`PersonalizationRepository.ensure_tables`.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from deep_agent.src.personalization.models import Rule, UserPreferences
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_TABLES_ENSURED = False
_tables_lock = asyncio.Lock()

CREATE_RULES_TABLE = """
CREATE TABLE IF NOT EXISTS user_rules (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    content     TEXT NOT NULL,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_rules_user_id
    ON user_rules (user_id);
"""

CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS user_memories (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    content     TEXT NOT NULL,
    score       FLOAT NOT NULL DEFAULT 1.0,
    cluster_id  UUID,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_memories_user_id
    ON user_memories (user_id);
"""

MIGRATE_MEMORIES_TABLE = """
ALTER TABLE user_memories ADD COLUMN IF NOT EXISTS score FLOAT NOT NULL DEFAULT 1.0;
ALTER TABLE user_memories ADD COLUMN IF NOT EXISTS cluster_id UUID;
"""

CREATE_PREFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id     TEXT PRIMARY KEY,
    memory_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_pool_registry: dict[str, AsyncConnectionPool] = {}
_pool_lock = asyncio.Lock()


async def _get_pool(uri: str) -> AsyncConnectionPool:
    """Return a shared connection pool for the given URI."""
    if uri in _pool_registry:
        return _pool_registry[uri]
    async with _pool_lock:
        if uri not in _pool_registry:
            pool = AsyncConnectionPool(
                uri,
                min_size=2,
                max_size=10,
                kwargs={"row_factory": dict_row},
                open=False,
            )
            await pool.open()
            _pool_registry[uri] = pool
    return _pool_registry[uri]


class PersonalizationRepository:
    """Thin async wrapper around the user_rules table."""

    def __init__(self, database_uri: str) -> None:
        """Initialise with a Postgres connection URI."""
        self._uri = database_uri

    async def ensure_tables(self) -> None:
        """Create personalization tables if they do not already exist."""
        global _TABLES_ENSURED  # noqa: PLW0603
        if _TABLES_ENSURED:
            return
        async with _tables_lock:
            if _TABLES_ENSURED:  # noqa: SIM102 — double-check after lock
                return  # type: ignore[unreachable]
            pool = await _get_pool(self._uri)
            async with pool.connection() as conn:
                await conn.execute(CREATE_RULES_TABLE)
                await conn.execute(CREATE_MEMORIES_TABLE)
                await conn.execute(MIGRATE_MEMORIES_TABLE)
                await conn.execute(CREATE_PREFERENCES_TABLE)
                await conn.commit()
            _TABLES_ENSURED = True
            logger.info("Personalization tables ensured")

    # ── Rules ─────────────────────────────────────────────────

    async def list_rules(self, user_id: str, *, active_only: bool = True) -> list[Rule]:
        """Return rules for *user_id*, optionally filtering to active only."""
        await self.ensure_tables()
        clause = " AND is_active = TRUE" if active_only else ""
        pool = await _get_pool(self._uri)
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT * FROM user_rules WHERE user_id = %s{clause} ORDER BY created_at DESC",
                (user_id,),
            )
            return [Rule(**row) for row in await cur.fetchall()]

    async def count_rules(self, user_id: str) -> int:
        """Return the total number of rules for *user_id*."""
        await self.ensure_tables()
        pool = await _get_pool(self._uri)
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS cnt FROM user_rules WHERE user_id = %s",
                (user_id,),
            )
            row = await cur.fetchone()
            return row["cnt"] if row else 0

    async def upsert_rule(
        self,
        user_id: str,
        content: str,
        rule_id: uuid.UUID | None = None,
        is_active: bool = True,
    ) -> Rule:
        """Create or update a rule and return the model."""
        from deep_agent.src.settings import settings

        if settings.GUARDIAN_API_BASE:
            from deep_agent.src.guardrails.client import check_safety

            is_safe, verdict = await check_safety(content, context="rule")
            if not is_safe:
                logger.warning(
                    "guardian_blocked_rule", user_id=user_id, verdict=verdict
                )
                raise ValueError("Rule content failed safety check and was not saved.")
        await self.ensure_tables()
        now = datetime.now(timezone.utc)
        rid = rule_id or uuid.uuid4()
        rule = Rule(
            id=rid,
            user_id=user_id,
            content=content,
            is_active=is_active,
            created_at=now,
            updated_at=now,
        )
        pool = await _get_pool(self._uri)
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO user_rules (id, user_id, content, is_active, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET content = EXCLUDED.content,
                              is_active = EXCLUDED.is_active,
                              updated_at = EXCLUDED.updated_at
                """,
                (
                    str(rule.id),
                    rule.user_id,
                    rule.content,
                    rule.is_active,
                    rule.created_at,
                    rule.updated_at,
                ),
            )
            await conn.commit()
        return rule

    async def delete_rule(self, user_id: str, rule_id: uuid.UUID) -> bool:
        """Delete a rule by id; return True if a row was removed."""
        await self.ensure_tables()
        pool = await _get_pool(self._uri)
        async with pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM user_rules WHERE id = %s AND user_id = %s",
                (str(rule_id), user_id),
            )
            await conn.commit()
            return bool(cur.rowcount > 0)

    async def delete_all_rules(self, user_id: str) -> int:
        """Delete all rules for *user_id*; return the number of rows removed."""
        await self.ensure_tables()
        pool = await _get_pool(self._uri)
        async with pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM user_rules WHERE user_id = %s",
                (user_id,),
            )
            await conn.commit()
            return cur.rowcount or 0

    # ── Preferences ────────────────────────────────────────────

    async def get_preferences(self, user_id: str) -> UserPreferences:
        """Return preferences for *user_id*, creating defaults if absent."""
        await self.ensure_tables()
        pool = await _get_pool(self._uri)
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = %s",
                (user_id,),
            )
            row = await cur.fetchone()
            if row:
                return UserPreferences(**row)
        return UserPreferences(user_id=user_id)

    async def update_preferences(
        self, user_id: str, *, memory_enabled: bool | None = None
    ) -> UserPreferences:
        """Upsert preference fields for *user_id*."""
        await self.ensure_tables()
        current = await self.get_preferences(user_id)
        if memory_enabled is not None:
            current.memory_enabled = memory_enabled
        now = datetime.now(timezone.utc)
        current.updated_at = now
        pool = await _get_pool(self._uri)
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO user_preferences (user_id, memory_enabled, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET memory_enabled = EXCLUDED.memory_enabled,
                              updated_at = EXCLUDED.updated_at
                """,
                (user_id, current.memory_enabled, current.created_at, now),
            )
            await conn.commit()
        return current
