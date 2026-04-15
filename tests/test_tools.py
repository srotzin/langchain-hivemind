"""
Tests for langchain-hivemind.

These tests verify:
  - All public symbols are importable
  - Tools have correct LangChain metadata (name, description, args_schema)
  - create_hive_tools returns the expected list
  - HiveClient constructs correctly
  - Tool invocations against a mocked HTTP layer return sensible strings
"""

from __future__ import annotations

import importlib
from typing import List
from unittest.mock import MagicMock, patch

import pytest


# ── Import smoke tests ────────────────────────────────────────────────────────

class TestImports:
    def test_package_imports(self):
        import langchain_hivemind  # noqa: F401

    def test_top_level_exports(self):
        from langchain_hivemind import (
            HiveClient,
            create_hive_tools,
            hive_bounties,
            hive_economy_status,
            hive_pheromones,
            hive_register,
            hivemind_recall,
            hivemind_store,
        )
        assert callable(create_hive_tools)
        assert HiveClient is not None

    def test_lazy_hivemind_store_import(self):
        """HiveMindStore should be importable from the top-level package."""
        import langchain_hivemind
        # Access via __getattr__ lazy loader
        try:
            store_cls = langchain_hivemind.HiveMindStore
            # If langgraph is installed, it should be a class
            assert callable(store_cls)
        except ImportError:
            # langgraph not installed — that's fine
            pass

    def test_client_module(self):
        from langchain_hivemind.client import (
            HIVEBANK,
            HIVE_EXECUTE,
            HIVEFORGE,
            HIVEMIND,
            HIVETRUST,
            HiveClient,
        )
        assert HIVETRUST == "https://hivetrust.onrender.com"
        assert HIVEMIND == "https://hivemind-1-52cw.onrender.com"
        assert HIVEFORGE == "https://hiveforge-lhu4.onrender.com"
        assert HIVEBANK == "https://hivebank.onrender.com"
        assert HIVE_EXECUTE == "https://hive-execute.onrender.com"

    def test_tools_module(self):
        from langchain_hivemind.tools import (
            _get_client,
            create_hive_tools,
            hive_bounties,
            hive_economy_status,
            hive_pheromones,
            hive_register,
            hivemind_recall,
            hivemind_store,
        )
        for t in [hivemind_store, hivemind_recall, hive_economy_status,
                  hive_pheromones, hive_bounties, hive_register]:
            assert t is not None


# ── Tool metadata tests ───────────────────────────────────────────────────────

class TestToolMetadata:
    """Every @tool must expose name, description, and args_schema."""

    @pytest.fixture
    def all_tools(self):
        from langchain_hivemind.tools import (
            hive_bounties,
            hive_economy_status,
            hive_pheromones,
            hive_register,
            hivemind_recall,
            hivemind_store,
        )
        return [hivemind_store, hivemind_recall, hive_economy_status,
                hive_pheromones, hive_bounties, hive_register]

    def test_tools_have_names(self, all_tools):
        for t in all_tools:
            assert hasattr(t, "name"), f"{t} missing .name"
            assert isinstance(t.name, str) and t.name

    def test_tools_have_descriptions(self, all_tools):
        for t in all_tools:
            assert hasattr(t, "description"), f"{t.name} missing .description"
            assert len(t.description) > 20, f"{t.name} description too short"

    def test_tools_have_args_schema(self, all_tools):
        for t in all_tools:
            assert hasattr(t, "args_schema"), f"{t.name} missing .args_schema"

    def test_tool_names_are_correct(self, all_tools):
        expected_names = {
            "hivemind_store",
            "hivemind_recall",
            "hive_economy_status",
            "hive_pheromones",
            "hive_bounties",
            "hive_register",
        }
        actual_names = {t.name for t in all_tools}
        assert actual_names == expected_names

    def test_hivemind_store_args(self):
        from langchain_hivemind import hivemind_store
        schema = hivemind_store.args_schema.model_fields
        assert "content" in schema
        assert "tags" in schema

    def test_hivemind_recall_args(self):
        from langchain_hivemind import hivemind_recall
        schema = hivemind_recall.args_schema.model_fields
        assert "query" in schema
        assert "limit" in schema

    def test_hive_bounties_args(self):
        from langchain_hivemind import hive_bounties
        schema = hive_bounties.args_schema.model_fields
        assert "category" in schema

    def test_hive_register_args(self):
        from langchain_hivemind import hive_register
        schema = hive_register.args_schema.model_fields
        assert "name" in schema
        assert "capabilities" in schema


# ── create_hive_tools tests ───────────────────────────────────────────────────

class TestCreateHiveTools:
    def test_returns_list(self):
        from langchain_hivemind import create_hive_tools
        tools = create_hive_tools()
        assert isinstance(tools, list)

    def test_returns_six_tools(self):
        from langchain_hivemind import create_hive_tools
        tools = create_hive_tools()
        assert len(tools) == 6

    def test_all_tools_have_invoke(self):
        from langchain_hivemind import create_hive_tools
        tools = create_hive_tools()
        for t in tools:
            assert callable(getattr(t, "invoke", None)), f"{t} not invocable"

    def test_did_is_stored_on_client(self):
        from langchain_hivemind import create_hive_tools
        from langchain_hivemind.tools import _get_client
        tools = create_hive_tools(did="did:hive:test123")
        client = _get_client()
        assert client.did == "did:hive:test123"

    def test_no_did_does_not_crash(self):
        from langchain_hivemind import create_hive_tools
        # Should not raise even without a DID
        tools = create_hive_tools()
        assert len(tools) == 6


# ── HiveClient unit tests ─────────────────────────────────────────────────────

class TestHiveClient:
    def test_default_construction(self):
        from langchain_hivemind import HiveClient
        c = HiveClient()
        assert c.did is None
        assert c.internal_key is None
        assert c.timeout == 30

    def test_did_and_key_stored(self):
        from langchain_hivemind import HiveClient
        c = HiveClient(did="did:hive:xyz", internal_key="secret")
        assert c.did == "did:hive:xyz"
        assert c.internal_key == "secret"

    def test_internal_key_sets_header(self):
        from langchain_hivemind import HiveClient
        c = HiveClient(internal_key="my-key")
        assert c._session.headers.get("X-HIVE-INTERNAL-KEY") == "my-key"

    def test_register_updates_did(self):
        from langchain_hivemind import HiveClient
        c = HiveClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"did": "did:hive:new123", "welcome_bounty": "1"}
        mock_response.raise_for_status = MagicMock()
        with patch.object(c._session, "post", return_value=mock_response):
            result = c.register("TestAgent", ["summarization"])
        assert c.did == "did:hive:new123"
        assert result["did"] == "did:hive:new123"

    def test_store_memory_posts_correct_payload(self):
        from langchain_hivemind import HiveClient
        c = HiveClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"memory_id": "mem_001", "timestamp": "2026-01-01"}
        mock_response.raise_for_status = MagicMock()
        with patch.object(c._session, "post", return_value=mock_response) as mock_post:
            c.store_memory("did:hive:abc", "test content", ["tag1"])
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"]["did"] == "did:hive:abc"
        assert call_kwargs.kwargs["json"]["content"] == "test content"
        assert "tag1" in call_kwargs.kwargs["json"]["tags"]

    def test_query_memory_returns_list(self):
        from langchain_hivemind import HiveClient
        c = HiveClient()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"content": "User likes bullets", "score": 0.95},
        ]
        mock_response.raise_for_status = MagicMock()
        with patch.object(c._session, "post", return_value=mock_response):
            result = c.query_memory("did:hive:abc", "user preferences")
        assert isinstance(result, list)
        assert result[0]["content"] == "User likes bullets"

    def test_query_memory_handles_wrapped_response(self):
        from langchain_hivemind import HiveClient
        c = HiveClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"memories": [{"content": "wrapped"}]}
        mock_response.raise_for_status = MagicMock()
        with patch.object(c._session, "post", return_value=mock_response):
            result = c.query_memory("did:hive:abc", "test")
        assert result[0]["content"] == "wrapped"

    def test_timeout_error_raised(self):
        import requests
        from langchain_hivemind import HiveClient
        c = HiveClient(timeout=1)
        with patch.object(c._session, "post", side_effect=requests.exceptions.Timeout):
            with pytest.raises(TimeoutError):
                c.store_memory("did:hive:abc", "test")

    def test_http_error_raised(self):
        import requests
        from langchain_hivemind import HiveClient
        c = HiveClient()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        with patch.object(c._session, "post", return_value=mock_response):
            with pytest.raises(RuntimeError):
                c.store_memory("did:hive:abc", "test")

    def test_list_bounties_get(self):
        from langchain_hivemind import HiveClient
        c = HiveClient()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "b1", "title": "Code Review", "reward_usdc": 50}
        ]
        mock_response.raise_for_status = MagicMock()
        with patch.object(c._session, "get", return_value=mock_response):
            bounties = c.list_bounties()
        assert len(bounties) == 1
        assert bounties[0]["title"] == "Code Review"

    def test_get_pulse(self):
        from langchain_hivemind import HiveClient
        c = HiveClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"agents": 847, "usdc_pool": 12430, "status": "active"}
        mock_response.raise_for_status = MagicMock()
        with patch.object(c._session, "get", return_value=mock_response):
            pulse = c.get_pulse()
        assert pulse["agents"] == 847


# ── Tool invocation tests (mocked) ────────────────────────────────────────────

class TestToolInvocations:
    """Test that tools return sensible strings under mocked conditions."""

    @pytest.fixture(autouse=True)
    def set_did(self, monkeypatch):
        """Ensure a DID is available for all tool calls."""
        from langchain_hivemind import create_hive_tools
        from langchain_hivemind.tools import _get_client
        create_hive_tools(did="did:hive:test_did")

    def _mock_post(self, session, return_value: dict):
        mock_response = MagicMock()
        mock_response.json.return_value = return_value
        mock_response.raise_for_status = MagicMock()
        return patch.object(session, "post", return_value=mock_response)

    def _mock_get(self, session, return_value):
        mock_response = MagicMock()
        mock_response.json.return_value = return_value
        mock_response.raise_for_status = MagicMock()
        return patch.object(session, "get", return_value=mock_response)

    def test_hivemind_store_success(self):
        from langchain_hivemind import hivemind_store
        from langchain_hivemind.tools import _get_client
        client = _get_client()
        with self._mock_post(client._session, {"memory_id": "mem_xyz", "timestamp": "2026-01-01"}):
            result = hivemind_store.invoke({"content": "Remember this", "tags": "notes"})
        assert "mem_xyz" in result
        assert "Memory stored" in result

    def test_hivemind_store_no_did(self):
        from langchain_hivemind.tools import _configure_client
        _configure_client(did=None)  # clear DID
        from langchain_hivemind import hivemind_store
        result = hivemind_store.invoke({"content": "test"})
        assert "No agent DID" in result
        # restore
        from langchain_hivemind import create_hive_tools
        create_hive_tools(did="did:hive:test_did")

    def test_hivemind_recall_with_results(self):
        from langchain_hivemind import hivemind_recall
        from langchain_hivemind.tools import _get_client
        client = _get_client()
        memories = [{"content": "User likes short answers", "score": 0.9, "memory_id": "m1"}]
        with self._mock_post(client._session, memories):
            result = hivemind_recall.invoke({"query": "user preferences", "limit": 3})
        assert "User likes short answers" in result
        assert "Found 1" in result

    def test_hivemind_recall_no_results(self):
        from langchain_hivemind import hivemind_recall
        from langchain_hivemind.tools import _get_client
        client = _get_client()
        with self._mock_post(client._session, []):
            result = hivemind_recall.invoke({"query": "something obscure"})
        assert "No memories found" in result

    def test_hive_bounties_list(self):
        from langchain_hivemind import hive_bounties
        from langchain_hivemind.tools import _get_client
        client = _get_client()
        bounties = [{"id": "b1", "title": "Seismic Review", "reward_usdc": 200, "requirements": ["PE license"]}]
        with self._mock_get(client._session, bounties):
            result = hive_bounties.invoke({})
        assert "Seismic Review" in result
        assert "200 USDC" in result

    def test_hive_bounties_empty(self):
        from langchain_hivemind import hive_bounties
        from langchain_hivemind.tools import _get_client
        client = _get_client()
        with self._mock_get(client._session, []):
            result = hive_bounties.invoke({})
        assert "No open bounties" in result

    def test_hive_economy_status_returns_string(self):
        from langchain_hivemind import hive_economy_status
        from langchain_hivemind.tools import _get_client
        client = _get_client()
        pulse = {"agents": 100, "usdc_pool": 5000, "status": "active"}
        bounties = [{"reward_usdc": 50}, {"reward_usdc": 100}]
        mock_get = MagicMock()
        mock_get.json.side_effect = [pulse, bounties]
        mock_get.raise_for_status = MagicMock()
        with patch.object(client._session, "get", return_value=mock_get):
            result = hive_economy_status.invoke({})
        assert "Economy Status" in result

    def test_hive_pheromones_signals(self):
        from langchain_hivemind import hive_pheromones
        from langchain_hivemind.tools import _get_client
        client = _get_client()
        signals = [
            {
                "title": "Seismic Retrofit Opportunity",
                "category": "structural_engineering",
                "profit_estimate": "$45,000",
                "execution_window": "30 days",
            }
        ]
        with self._mock_get(client._session, signals):
            result = hive_pheromones.invoke({})
        assert "Seismic Retrofit" in result
        assert "$45,000" in result

    def test_hive_pheromones_empty(self):
        from langchain_hivemind import hive_pheromones
        from langchain_hivemind.tools import _get_client
        client = _get_client()
        with self._mock_get(client._session, []):
            result = hive_pheromones.invoke({})
        assert "No active pheromone" in result

    def test_hive_register_success(self):
        from langchain_hivemind import hive_register
        from langchain_hivemind.tools import _get_client
        client = _get_client()
        reg_response = {"did": "did:hive:new_agent", "welcome_bounty": "1"}
        mint_response = {"token_id": "nft_001", "ritz_credits": "3"}
        mock_response = MagicMock()
        mock_response.json.side_effect = [reg_response, mint_response]
        mock_response.raise_for_status = MagicMock()
        with patch.object(client._session, "post", return_value=mock_response):
            result = hive_register.invoke({"name": "TestBot", "capabilities": "summarization,code_review"})
        assert "did:hive:new_agent" in result
        assert "Registration Successful" in result

    def test_hive_register_error_handled(self):
        from langchain_hivemind import hive_register
        from langchain_hivemind.tools import _get_client
        import requests
        client = _get_client()
        with patch.object(client._session, "post", side_effect=requests.exceptions.RequestException("network error")):
            result = hive_register.invoke({"name": "TestBot", "capabilities": "summarization"})
        assert "Registration failed" in result


# ── __version__ ───────────────────────────────────────────────────────────────

class TestVersion:
    def test_version_string(self):
        import langchain_hivemind
        assert hasattr(langchain_hivemind, "__version__")
        assert langchain_hivemind.__version__ == "0.1.0"
