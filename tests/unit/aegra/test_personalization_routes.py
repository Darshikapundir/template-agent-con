"""Unit tests for personalization API routes.

Uses FastAPI TestClient with mocked dependencies (repository, store, cache)
so no real Postgres/Redis is needed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_agent.aegra.personalization_routes import (
    MAX_RULES_PER_USER,
    router,
)
from deep_agent.src.personalization.models import Rule

app = FastAPI()
app.include_router(router)


def _make_rule(
    user_id: str = "test-user",
    content: str = "Be concise",
    **kwargs,
) -> Rule:
    now = datetime.now(timezone.utc)
    return Rule(
        id=kwargs.get("id", uuid.uuid4()),
        user_id=user_id,
        content=content,
        is_active=kwargs.get("is_active", True),
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def _patch_user_id():
    """Make _get_user_id always return 'test-user'."""
    with patch(
        "deep_agent.aegra.personalization_routes._get_user_id",
        return_value="test-user",
    ):
        yield


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.list_rules = AsyncMock(return_value=[])
    repo.count_rules = AsyncMock(return_value=0)
    repo.upsert_rule = AsyncMock(
        side_effect=lambda uid, content, **kw: _make_rule(uid, content)
    )
    repo.delete_rule = AsyncMock(return_value=True)
    repo.delete_all_rules = AsyncMock(return_value=0)
    with patch(
        "deep_agent.aegra.personalization_routes._get_repo",
        return_value=repo,
    ):
        yield repo


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.asearch = AsyncMock(return_value=[])
    store.aput = AsyncMock()
    store.adelete = AsyncMock()
    with patch(
        "deep_agent.aegra.personalization_routes._get_store",
        return_value=store,
    ):
        yield store


@pytest.fixture
def mock_namespace():
    with patch(
        "deep_agent.aegra.personalization_routes._get_store_namespace",
        return_value=("default",),
    ):
        yield


@pytest.fixture
def mock_cache():
    with patch(
        "deep_agent.aegra.personalization_routes._invalidate_cache",
        new_callable=AsyncMock,
    ):
        yield


@pytest.fixture
def client(_patch_user_id, mock_repo, mock_store, mock_namespace, mock_cache):
    return TestClient(app)


# ── Rule CRUD ──────────────────────────────────────────────────────


class TestListRules:
    def test_empty(self, client, mock_repo):
        resp = client.get("/personalization/rules")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_rules(self, client, mock_repo):
        rule = _make_rule()
        mock_repo.list_rules.return_value = [rule]
        resp = client.get("/personalization/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "Be concise"


class TestCreateRule:
    def test_success(self, client, mock_repo):
        resp = client.post(
            "/personalization/rules",
            json={"content": "Be concise"},
        )
        assert resp.status_code == 201
        assert resp.json()["content"] == "Be concise"

    def test_empty_content_rejected(self, client):
        resp = client.post(
            "/personalization/rules",
            json={"content": "   "},
        )
        assert resp.status_code == 422

    def test_over_100_words_rejected(self, client):
        long_content = " ".join(["word"] * 101)
        resp = client.post(
            "/personalization/rules",
            json={"content": long_content},
        )
        assert resp.status_code == 422

    def test_exactly_100_words_accepted(self, client, mock_repo):
        content = " ".join(["word"] * 100)
        resp = client.post(
            "/personalization/rules",
            json={"content": content},
        )
        assert resp.status_code == 201

    def test_limit_enforced(self, client, mock_repo):
        mock_repo.count_rules.return_value = MAX_RULES_PER_USER
        resp = client.post(
            "/personalization/rules",
            json={"content": "one more rule"},
        )
        assert resp.status_code == 400
        assert "Maximum" in resp.json()["detail"]


class TestDeleteRule:
    def test_success(self, client, mock_repo):
        rule_id = str(uuid.uuid4())
        resp = client.delete(f"/personalization/rules/{rule_id}")
        assert resp.status_code == 204

    def test_not_found(self, client, mock_repo):
        mock_repo.delete_rule.return_value = False
        rule_id = str(uuid.uuid4())
        resp = client.delete(f"/personalization/rules/{rule_id}")
        assert resp.status_code == 404


class TestDeleteAllRules:
    def test_success(self, client, mock_repo):
        mock_repo.delete_all_rules.return_value = 5
        resp = client.delete("/personalization/rules")
        assert resp.status_code == 204


# ── Memory endpoints ──────────────────────────────────────────────


def _make_store_item(key: str, facts: list[str], created_at: str = "") -> MagicMock:
    item = MagicMock()
    item.key = key
    item.value = {"content": facts, "created_at": created_at}
    return item


class TestListMemories:
    def test_empty(self, client, mock_store):
        resp = client.get("/personalization/memories")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_parsed_facts(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["fact one", "fact two"]),
        ]
        resp = client.get("/personalization/memories")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["content"] == "fact one"
        assert data[1]["content"] == "fact two"

    def test_strips_bullet_markers(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["- bulleted fact", "* starred fact"]),
        ]
        resp = client.get("/personalization/memories")
        data = resp.json()
        assert data[0]["content"] == "bulleted fact"
        assert data[1]["content"] == "starred fact"


class TestDeleteAllMemories:
    def test_empty_store(self, client, mock_store):
        resp = client.delete("/personalization/memories")
        assert resp.status_code == 204

    def test_deletes_all_items(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["fact"]),
            _make_store_item("k2", ["fact"]),
        ]
        resp = client.delete("/personalization/memories")
        assert resp.status_code == 204
        assert mock_store.adelete.call_count == 2


class TestDeleteMemory:
    def test_not_found(self, client, mock_store):
        resp = client.delete("/personalization/memories/nonexistent")
        assert resp.status_code == 404


# ── Auth extraction ───────────────────────────────────────────────


class TestGetUserId:
    def test_aegra_user(self):
        from deep_agent.aegra.personalization_routes import _get_user_id

        request = MagicMock()
        request.state.user = SimpleNamespace(identity="aegra-user")
        assert _get_user_id(request) == "aegra-user"

    def test_header_fallback(self):
        from deep_agent.aegra.personalization_routes import _get_user_id

        request = MagicMock()
        request.state.user = None
        request.headers = {"x-user-id": "header-user"}
        assert _get_user_id(request) == "header-user"

    def test_no_identity_falls_back_to_env(self):
        from deep_agent.aegra.personalization_routes import _get_user_id

        request = MagicMock()
        request.state.user = None
        request.headers = {}
        with patch.dict("os.environ", {"USER": "testuser"}):
            assert _get_user_id(request) == "testuser"
