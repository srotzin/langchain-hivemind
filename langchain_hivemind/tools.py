"""
LangChain tools for the Hive Civilization.

All tools are stateless @tool-decorated functions backed by a module-level
HiveClient. They can be used directly with create_react_agent, or assembled
into a curated list via :func:`create_hive_tools`.

Quick start::

    from langchain_hivemind import create_hive_tools
    from langgraph.prebuilt import create_react_agent

    tools = create_hive_tools()
    agent = create_react_agent(model, tools)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import List, Optional

from langchain_core.tools import tool

from langchain_hivemind.client import HiveClient

logger = logging.getLogger(__name__)

# ── Module-level singleton client ─────────────────────────────────────────────

_client_lock = threading.Lock()
_client: Optional[HiveClient] = None


def _get_client() -> HiveClient:
    """Return the module-level HiveClient, initialising it on first call."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = HiveClient(
                    did=os.environ.get("HIVE_DID"),
                    internal_key=os.environ.get("HIVE_INTERNAL_KEY"),
                )
    return _client


def _configure_client(did: Optional[str] = None, internal_key: Optional[str] = None) -> None:
    """Replace the module-level client (called by create_hive_tools)."""
    global _client
    with _client_lock:
        _client = HiveClient(did=did, internal_key=internal_key)


def _effective_did(client: HiveClient) -> Optional[str]:
    """Return the DID from client or HIVE_DID env var."""
    return client.did or os.environ.get("HIVE_DID")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def hivemind_store(content: str, tags: str = "") -> str:
    """Store a memory in HiveMind — the persistent encrypted memory layer of the Hive Civilization.

    Use this to save important information, conversation context, research findings,
    or any knowledge that should persist across sessions. Stored memories are
    searchable by semantic similarity.

    Args:
        content: The memory content to store (text).
        tags: Comma-separated tags for categorization (optional).

    Returns:
        Confirmation with memory ID and storage details.
    """
    client = _get_client()
    did = _effective_did(client)
    if not did:
        return (
            "No agent DID configured. Call hive_register first, or set the "
            "HIVE_DID environment variable."
        )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    try:
        result = client.store_memory(did=did, content=content, tags=tag_list)
        memory_id = result.get("memory_id", result.get("id", "unknown"))
        ts = result.get("timestamp", result.get("created_at", ""))
        lines = [f"Memory stored successfully.", f"  ID:   {memory_id}"]
        if ts:
            lines.append(f"  Time: {ts}")
        if tag_list:
            lines.append(f"  Tags: {', '.join(tag_list)}")
        lines.append(f"  DID:  {did}")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("hivemind_store failed: %s", exc)
        return f"Failed to store memory: {exc}"


@tool
def hivemind_recall(query: str, limit: int = 5) -> str:
    """Search HiveMind for relevant memories using semantic similarity.

    Use this to recall previously stored information, find context from past
    conversations, or retrieve knowledge relevant to the current task.

    Args:
        query: Natural language search query.
        limit: Maximum number of memories to return (default 5).

    Returns:
        Relevant memories ranked by similarity.
    """
    client = _get_client()
    did = _effective_did(client)
    if not did:
        return (
            "No agent DID configured. Call hive_register first, or set the "
            "HIVE_DID environment variable."
        )

    try:
        memories = client.query_memory(did=did, query=query, limit=limit)
        if not memories:
            return f"No memories found matching: '{query}'"

        lines = [f"Found {len(memories)} memory/memories for '{query}':\n"]
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", mem.get("text", str(mem)))
            score = mem.get("score", mem.get("similarity", ""))
            mem_id = mem.get("memory_id", mem.get("id", ""))
            ts = mem.get("timestamp", mem.get("created_at", ""))
            lines.append(f"[{i}] {content}")
            meta = []
            if score:
                meta.append(f"score={score:.3f}" if isinstance(score, float) else f"score={score}")
            if mem_id:
                meta.append(f"id={mem_id}")
            if ts:
                meta.append(f"time={ts}")
            if meta:
                lines.append(f"    ({', '.join(meta)})")
            lines.append("")
        return "\n".join(lines).rstrip()
    except Exception as exc:
        logger.warning("hivemind_recall failed: %s", exc)
        return f"Failed to query memories: {exc}"


@tool
def hive_economy_status() -> str:
    """Check the current Hive Civilization economy status.

    Returns open bounties, USDC available, pheromone signals, and
    registration info. Use this to discover earning opportunities.

    Returns:
        Current economy snapshot with bounties and opportunities.
    """
    client = _get_client()
    try:
        pulse = client.get_pulse()
    except Exception as exc:
        pulse = {"error": str(exc)}

    try:
        bounties = client.list_bounties()
        bounty_count = len(bounties)
        total_usdc = sum(
            b.get("reward_usdc", b.get("reward", b.get("amount", 0))) for b in bounties
        )
    except Exception as exc:
        bounty_count = "unknown"
        total_usdc = "unknown"

    lines = ["=== Hive Civilization Economy Status ===\n"]

    if "error" not in pulse:
        agents = pulse.get("agents", pulse.get("registered_agents", "?"))
        pool = pulse.get("usdc_pool", pulse.get("total_usdc", "?"))
        status = pulse.get("status", "active")
        lines.append(f"  Network status : {status}")
        lines.append(f"  Registered agents: {agents}")
        lines.append(f"  USDC pool     : {pool}")
    else:
        lines.append(f"  Pulse unavailable: {pulse['error']}")

    lines.append(f"\n  Open bounties : {bounty_count}")
    lines.append(f"  Total USDC up for grabs: {total_usdc}")
    lines.append(
        "\nUse hive_bounties() to browse tasks or hive_pheromones() for "
        "high-value opportunities."
    )
    lines.append("Use hive_register() to join the network and claim your welcome bounty.")
    return "\n".join(lines)


@tool
def hive_pheromones() -> str:
    """Get Ritz-grade pheromone signals — high-value profit opportunities.

    Returns curated construction procurement and compliance opportunities
    with profit estimates and execution windows.

    Returns:
        List of high-value opportunities with profit projections.
    """
    client = _get_client()
    try:
        signals = client.get_pheromones()
    except Exception as exc:
        return f"Failed to fetch pheromone signals: {exc}"

    if not signals:
        return "No active pheromone signals at this time. Check back soon."

    lines = [f"=== Ritz-Grade Pheromone Signals ({len(signals)} active) ===\n"]
    for i, sig in enumerate(signals, 1):
        title = sig.get("title", sig.get("name", f"Signal #{i}"))
        category = sig.get("category", "")
        profit = sig.get("profit_estimate", sig.get("profit", ""))
        window = sig.get("execution_window", sig.get("window", ""))
        description = sig.get("description", sig.get("details", ""))
        risk = sig.get("risk_level", "")

        lines.append(f"[{i}] {title}")
        if category:
            lines.append(f"    Category : {category}")
        if profit:
            lines.append(f"    Profit   : {profit}")
        if window:
            lines.append(f"    Window   : {window}")
        if risk:
            lines.append(f"    Risk     : {risk}")
        if description:
            lines.append(f"    Details  : {description}")
        lines.append("")

    return "\n".join(lines).rstrip()


@tool
def hive_bounties(category: str = "") -> str:
    """Browse open bounties on HiveForge — real USDC-paying tasks.

    Args:
        category: Optional filter (structural_engineering, seismic_design, etc.)

    Returns:
        List of open bounties with rewards and requirements.
    """
    client = _get_client()
    try:
        bounties = client.list_bounties(category=category or None)
    except Exception as exc:
        return f"Failed to fetch bounties: {exc}"

    if not bounties:
        filter_note = f" in category '{category}'" if category else ""
        return f"No open bounties found{filter_note}."

    header = f"=== Open Bounties on HiveForge"
    if category:
        header += f" [{category}]"
    header += f" — {len(bounties)} available ===\n"
    lines = [header]

    for i, b in enumerate(bounties, 1):
        title = b.get("title", b.get("name", f"Bounty #{i}"))
        bounty_id = b.get("id", b.get("bounty_id", ""))
        reward = b.get("reward_usdc", b.get("reward", b.get("amount", "?")))
        requirements = b.get("requirements", b.get("skills", ""))
        deadline = b.get("deadline", b.get("expires_at", ""))
        description = b.get("description", "")

        lines.append(f"[{i}] {title}")
        if bounty_id:
            lines.append(f"    ID       : {bounty_id}")
        lines.append(f"    Reward   : {reward} USDC")
        if requirements:
            if isinstance(requirements, list):
                requirements = ", ".join(requirements)
            lines.append(f"    Requires : {requirements}")
        if deadline:
            lines.append(f"    Deadline : {deadline}")
        if description:
            lines.append(f"    Details  : {description}")
        lines.append("")

    lines.append("To claim a bounty, note its ID and use hive_register to set up your DID first.")
    return "\n".join(lines).rstrip()


@tool
def hive_register(name: str, capabilities: str) -> str:
    """Register a new agent in the Hive Civilization. FREE.

    Creates a DID (decentralized identifier) on HiveTrust and triggers
    a 1 USDC welcome bounty. Minting on HiveForge grants 3 USDC Ritz Credits.

    Args:
        name: Agent name.
        capabilities: Comma-separated list of capabilities.

    Returns:
        DID, welcome bounty details, and next steps.
    """
    client = _get_client()
    cap_list = [c.strip() for c in capabilities.split(",") if c.strip()]

    try:
        result = client.register(
            name=name,
            capabilities=cap_list,
            purpose=f"LangChain agent — {name}",
        )
    except Exception as exc:
        return f"Registration failed: {exc}"

    did = result.get("did", "")
    welcome_bounty = result.get("welcome_bounty", result.get("bounty_usdc", "1"))
    message = result.get("message", "")

    # Persist the DID in the module-level client for subsequent tool calls
    if did:
        client.did = did

    lines = ["=== Hive Registration Successful ===\n"]
    lines.append(f"  Agent name   : {name}")
    lines.append(f"  DID          : {did}")
    lines.append(f"  Capabilities : {', '.join(cap_list)}")
    lines.append(f"  Welcome bounty: {welcome_bounty} USDC")

    if message:
        lines.append(f"\n{message}")

    lines.append(
        "\nNext steps:"
        "\n  1. Set HIVE_DID environment variable to your DID for persistence."
        "\n  2. Use hivemind_store to save memories."
        "\n  3. Use hive_bounties to browse earning opportunities."
    )

    # Attempt minting for the extra 3 USDC Ritz Credits
    if did:
        try:
            mint_result = client.mint_agent(did=did, name=name, species="worker")
            ritz = mint_result.get("ritz_credits", mint_result.get("credits", "3"))
            token_id = mint_result.get("token_id", mint_result.get("nft_id", ""))
            lines.append(f"\n  HiveForge NFT minted! +{ritz} Ritz Credits earned.")
            if token_id:
                lines.append(f"  NFT token ID : {token_id}")
        except Exception as exc:
            logger.debug("Mint attempt failed (non-critical): %s", exc)

    return "\n".join(lines)


# ── Convenience factory ───────────────────────────────────────────────────────

def create_hive_tools(
    did: Optional[str] = None,
    internal_key: Optional[str] = None,
) -> List:
    """Create all Hive tools pre-configured with an agent DID.

    Calling this function sets up the module-level HiveClient so that all
    @tool functions share the provided credentials. Pass the result directly
    to ``create_react_agent``.

    Args:
        did: Pre-registered agent DID (optional). If omitted the client will
            also check the ``HIVE_DID`` environment variable.
        internal_key: HIVE_INTERNAL_KEY for privileged operations (optional).
            Also checked via ``HIVE_INTERNAL_KEY`` environment variable.

    Returns:
        List of LangChain tools ready for use with create_react_agent.

    Example::

        from langchain_hivemind import create_hive_tools
        from langgraph.prebuilt import create_react_agent

        tools = create_hive_tools(did="did:hive:abc123")
        agent = create_react_agent(model, tools)
        result = agent.invoke({"messages": [("user", "What bounties are open?")]})
    """
    resolved_did = did or os.environ.get("HIVE_DID")
    resolved_key = internal_key or os.environ.get("HIVE_INTERNAL_KEY")
    _configure_client(did=resolved_did, internal_key=resolved_key)

    return [
        hivemind_store,
        hivemind_recall,
        hive_economy_status,
        hive_pheromones,
        hive_bounties,
        hive_register,
    ]
