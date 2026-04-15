"""
langchain-hivemind
==================

LangChain / LangGraph integration for the Hive Civilization — persistent
encrypted memory, bounties, and USDC economy for AI agents.

Quick start::

    from langchain_hivemind import create_hive_tools
    from langgraph.prebuilt import create_react_agent

    tools = create_hive_tools()                       # or pass did="did:hive:..."
    agent = create_react_agent(model, tools)
    agent.invoke({"messages": [("user", "What bounties are open?")]})

Tools
-----
hivemind_store      — Persist a memory in HiveMind
hivemind_recall     — Search memories by semantic similarity
hive_economy_status — Live economy snapshot (bounties, USDC pool, agents)
hive_pheromones     — Ritz-grade high-value opportunity signals
hive_bounties       — Browse open USDC-paying bounties
hive_register       — Register a new agent (free, includes welcome bounty)

Low-level clients
-----------------
HiveClient          — Direct HTTP access to all Hive micro-services
HiveMindStore       — LangGraph BaseStore backed by HiveMind (optional dep)
"""

from langchain_hivemind.client import HiveClient
from langchain_hivemind.tools import (
    create_hive_tools,
    hive_bounties,
    hive_economy_status,
    hive_pheromones,
    hive_register,
    hivemind_recall,
    hivemind_store,
)

__version__ = "0.1.0"

__all__ = [
    # Tools
    "hivemind_store",
    "hivemind_recall",
    "hive_economy_status",
    "hive_pheromones",
    "hive_bounties",
    "hive_register",
    "create_hive_tools",
    # Client
    "HiveClient",
    # Store (import lazily to avoid hard langgraph dep)
    "HiveMindStore",
]


def __getattr__(name: str):
    if name == "HiveMindStore":
        from langchain_hivemind.memory_store import HiveMindStore  # noqa: PLC0415
        return HiveMindStore
    raise AttributeError(f"module 'langchain_hivemind' has no attribute {name!r}")
