"""
HiveMindStore — a LangGraph BaseStore implementation backed by HiveMind.

This enables first-class LangGraph cross-thread memory without any custom
persistence layer. Pass an instance to ``create_react_agent`` or use it
directly with any LangGraph workflow.

Example::

    from langchain_hivemind import HiveMindStore
    from langgraph.prebuilt import create_react_agent

    store = HiveMindStore(did="did:hive:abc123")
    agent = create_react_agent(model, tools, store=store)

Requires the optional ``langgraph`` dependency::

    pip install "langchain-hivemind[langgraph]"
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from langchain_hivemind.client import HiveClient

logger = logging.getLogger(__name__)

# ── LangGraph BaseStore ───────────────────────────────────────────────────────
# We import at runtime to avoid a hard dependency on langgraph when the
# optional extra is not installed.

try:
    from langgraph.store.base import BaseStore, Item, Op, Result, SearchItem
    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGGRAPH_AVAILABLE = False

    # Minimal shims so the module can still be imported and give a clear error
    class BaseStore:  # type: ignore[no-redef]
        pass

    Item = Dict  # type: ignore[misc, assignment]
    SearchItem = Dict  # type: ignore[misc, assignment]


class HiveMindStore(BaseStore):
    """LangGraph ``BaseStore`` powered by HiveMind persistent memory.

    The LangGraph ``put`` / ``get`` / ``search`` / ``delete`` contract is
    mapped onto HiveMind's ``/v1/store`` and ``/v1/query`` endpoints.
    Namespaces are serialised as tag prefixes so that memories stay
    logically separated.

    Args:
        did: Agent DID for all memory operations.
        client: Optional pre-built :class:`HiveClient`. If omitted, a default
            client is constructed.

    Raises:
        ImportError: If ``langgraph`` is not installed.
    """

    def __init__(
        self,
        did: str,
        client: Optional[HiveClient] = None,
    ) -> None:
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is required for HiveMindStore. "
                "Install it with: pip install 'langchain-hivemind[langgraph]'"
            )
        self.did = did
        self.client = client or HiveClient(did=did)
        # In-process key→memory_id cache for get/delete
        self._id_cache: Dict[str, str] = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ns_tag(namespace: Tuple[str, ...]) -> str:
        return "::".join(namespace)

    def _make_item(self, mem: Dict, namespace: Tuple[str, ...], key: str) -> "Item":
        """Convert a raw HiveMind memory dict into a LangGraph Item."""
        return Item(
            namespace=namespace,
            key=key,
            value=mem.get("content", mem.get("text", "")),
            created_at=mem.get("created_at", mem.get("timestamp", "")),
            updated_at=mem.get("updated_at", mem.get("timestamp", "")),
            score=mem.get("score"),
        )

    # ── LangGraph BaseStore interface ─────────────────────────────────────────

    def put(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        index: Optional[List[str]] = None,
    ) -> None:
        """Store an item in HiveMind under the given namespace/key."""
        ns_tag = self._ns_tag(namespace)
        content = value if isinstance(value, str) else str(value)
        tags = [ns_tag, f"key:{key}"]
        try:
            result = self.client.store_memory(did=self.did, content=content, tags=tags)
            memory_id = result.get("memory_id", result.get("id", ""))
            if memory_id:
                cache_key = f"{ns_tag}::{key}"
                self._id_cache[cache_key] = memory_id
        except Exception as exc:
            logger.error("HiveMindStore.put failed for %s/%s: %s", ns_tag, key, exc)
            raise

    def get(
        self,
        namespace: Tuple[str, ...],
        key: str,
    ) -> Optional["Item"]:
        """Retrieve a specific item by namespace and key."""
        ns_tag = self._ns_tag(namespace)
        query = f"key:{key} namespace:{ns_tag}"
        try:
            memories = self.client.query_memory(did=self.did, query=query, limit=1)
            if not memories:
                return None
            return self._make_item(memories[0], namespace, key)
        except Exception as exc:
            logger.error("HiveMindStore.get failed for %s/%s: %s", ns_tag, key, exc)
            return None

    def search(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List["SearchItem"]:
        """Search memories within a namespace prefix."""
        ns_tag = self._ns_tag(namespace_prefix)
        effective_query = query or ns_tag
        try:
            memories = self.client.query_memory(
                did=self.did,
                query=effective_query,
                limit=limit + offset,
            )
            sliced = memories[offset : offset + limit]
            return [
                self._make_item(mem, namespace_prefix, f"result_{i}")
                for i, mem in enumerate(sliced)
            ]
        except Exception as exc:
            logger.error("HiveMindStore.search failed for prefix %s: %s", ns_tag, exc)
            return []

    def delete(
        self,
        namespace: Tuple[str, ...],
        key: str,
    ) -> None:
        """Remove an item. HiveMind does not expose a delete endpoint yet;
        this is a no-op that logs a warning."""
        ns_tag = self._ns_tag(namespace)
        logger.warning(
            "HiveMindStore.delete called for %s/%s — HiveMind does not support "
            "deletion; memory will remain but be unfindable by key lookup.",
            ns_tag,
            key,
        )

    # ── Async stubs (delegate to sync for now) ────────────────────────────────

    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        index: Optional[List[str]] = None,
    ) -> None:
        """Async variant — delegates to synchronous put."""
        self.put(namespace, key, value, index=index)

    async def aget(
        self,
        namespace: Tuple[str, ...],
        key: str,
    ) -> Optional["Item"]:
        """Async variant — delegates to synchronous get."""
        return self.get(namespace, key)

    async def asearch(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List["SearchItem"]:
        """Async variant — delegates to synchronous search."""
        return self.search(
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )

    async def adelete(
        self,
        namespace: Tuple[str, ...],
        key: str,
    ) -> None:
        """Async variant — delegates to synchronous delete."""
        self.delete(namespace, key)

    def batch(self, ops: Iterable["Op"]) -> List["Result"]:
        """Execute a batch of operations sequentially."""
        results = []
        for op in ops:
            op_type = type(op).__name__
            try:
                if op_type == "PutOp":
                    self.put(op.namespace, op.key, op.value)
                    results.append(None)
                elif op_type == "GetOp":
                    results.append(self.get(op.namespace, op.key))
                elif op_type == "SearchOp":
                    results.append(
                        self.search(
                            op.namespace_prefix,
                            query=op.query,
                            filter=op.filter,
                            limit=op.limit,
                            offset=op.offset,
                        )
                    )
                elif op_type == "DeleteOp":
                    self.delete(op.namespace, op.key)
                    results.append(None)
                else:
                    logger.warning("Unknown op type: %s", op_type)
                    results.append(None)
            except Exception as exc:
                logger.error("Batch op %s failed: %s", op_type, exc)
                results.append(None)
        return results

    async def abatch(self, ops: Iterable["Op"]) -> List["Result"]:
        """Async batch — delegates to synchronous batch."""
        return self.batch(ops)
