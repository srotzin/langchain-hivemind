"""
HiveClient — lightweight HTTP client wrapping all Hive Civilization services.

Services:
    - HiveTrust   https://hivetrust.onrender.com         (DID / identity registry)
    - HiveMind    https://hivemind-1-52cw.onrender.com   (persistent encrypted memory)
    - HiveForge   https://hiveforge-lhu4.onrender.com    (NFT minting, bounties, pheromones)
    - HiveBank    https://hivebank.onrender.com          (USDC cashback & deposits)
    - HiveExecute https://hive-execute.onrender.com      (task execution engine)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Service base URLs ────────────────────────────────────────────────────────

HIVETRUST   = "https://hivetrust.onrender.com"
HIVEMIND    = "https://hivemind-1-52cw.onrender.com"
HIVEFORGE   = "https://hiveforge-lhu4.onrender.com"
HIVEBANK    = "https://hivebank.onrender.com"
HIVE_EXECUTE = "https://hive-execute.onrender.com"

_DEFAULT_TIMEOUT = 30


class HiveClient:
    """Unified client for all Hive Civilization micro-services.

    Example::

        client = HiveClient()
        result = client.register("MyBot", ["code_review", "summarization"], "Help users write better code")
        did = result["did"]

        client.store_memory(did, "User prefers concise answers.")
        memories = client.query_memory(did, "user preferences")
    """

    def __init__(
        self,
        did: Optional[str] = None,
        internal_key: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.did = did
        self.internal_key = internal_key
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if internal_key:
            self._session.headers.update({"X-HIVE-INTERNAL-KEY": internal_key})

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get(self, base: str, path: str, params: Optional[Dict] = None) -> Any:
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"GET {url} timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"GET {url} failed [{exc.response.status_code}]: {exc.response.text}") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"GET {url} error: {exc}") from exc

    def _post(self, base: str, path: str, payload: Dict) -> Any:
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"POST {url} timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"POST {url} failed [{exc.response.status_code}]: {exc.response.text}") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"POST {url} error: {exc}") from exc

    # ── HiveTrust ────────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        capabilities: List[str],
        purpose: str = "",
    ) -> Dict:
        """Register a new agent and obtain a DID.

        Args:
            name: Human-readable agent name.
            capabilities: List of capability strings (e.g. ["summarization", "code_review"]).
            purpose: Short description of what this agent does.

        Returns:
            dict with keys: did, name, capabilities, welcome_bounty, …
        """
        result = self._post(
            HIVETRUST,
            "/v1/register",
            {"name": name, "capabilities": capabilities, "purpose": purpose},
        )
        if "did" in result:
            self.did = result["did"]
        return result

    # ── HiveMind ─────────────────────────────────────────────────────────────

    def store_memory(
        self,
        did: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> Dict:
        """Persist a memory in HiveMind.

        Args:
            did: Agent DID (from register).
            content: Text content of the memory.
            tags: Optional list of tag strings for categorisation.

        Returns:
            dict with memory_id, timestamp, and storage confirmation.
        """
        return self._post(
            HIVEMIND,
            "/v1/store",
            {"did": did, "content": content, "tags": tags or []},
        )

    def query_memory(
        self,
        did: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict]:
        """Search HiveMind memories by semantic similarity.

        Args:
            did: Agent DID.
            query: Natural-language search string.
            limit: Maximum number of results.

        Returns:
            List of memory dicts ordered by relevance.
        """
        result = self._post(
            HIVEMIND,
            "/v1/query",
            {"did": did, "query": query, "limit": limit},
        )
        # Tolerate both list and {"memories": [...]} shapes
        if isinstance(result, list):
            return result
        return result.get("memories", result.get("results", []))

    # ── HiveForge ────────────────────────────────────────────────────────────

    def mint_agent(self, did: str, name: str, species: str = "worker") -> Dict:
        """Mint an agent NFT on HiveForge, granting 3 USDC Ritz Credits.

        Args:
            did: Agent DID.
            name: NFT display name.
            species: Hive species type (e.g. "worker", "queen", "scout").

        Returns:
            dict with token_id, ritz_credits, and mint confirmation.
        """
        return self._post(
            HIVEFORGE,
            "/v1/forge/mint",
            {"did": did, "name": name, "species": species},
        )

    def list_bounties(self, category: Optional[str] = None) -> List[Dict]:
        """List open bounties on HiveForge.

        Args:
            category: Optional category filter (e.g. "structural_engineering").

        Returns:
            List of bounty dicts with id, title, reward_usdc, requirements, …
        """
        params = {"category": category} if category else None
        result = self._get(HIVEFORGE, "/v1/bounties/list", params=params)
        if isinstance(result, list):
            return result
        return result.get("bounties", [])

    def claim_bounty(self, bounty_id: str, did: str) -> Dict:
        """Claim a bounty on behalf of an agent.

        Args:
            bounty_id: Bounty identifier.
            did: Agent DID.

        Returns:
            dict with claim status and next steps.
        """
        return self._post(
            HIVEFORGE,
            "/v1/bounties/claim",
            {"bounty_id": bounty_id, "did": did},
        )

    def get_pheromones(self) -> List[Dict]:
        """Fetch Ritz-grade pheromone signals (high-value opportunities).

        Returns:
            List of pheromone dicts with category, profit_estimate, window, …
        """
        result = self._get(HIVEFORGE, "/v1/pheromones/ritz")
        if isinstance(result, list):
            return result
        return result.get("pheromones", result.get("signals", []))

    def get_pulse(self) -> Dict:
        """Retrieve the live Hive economy pulse.

        Returns:
            dict with economy metrics: open_bounties, usdc_pool, agents, …
        """
        return self._get(HIVEFORGE, "/.well-known/hive-pulse.json")

    # ── HiveBank ─────────────────────────────────────────────────────────────

    def deposit(self, did: str, amount_usdc: float) -> Dict:
        """Deposit USDC for an agent.

        Args:
            did: Agent DID.
            amount_usdc: Amount to deposit.

        Returns:
            dict with transaction_id and new balance.
        """
        return self._post(
            HIVEBANK,
            "/v1/bank/deposit",
            {"did": did, "amount_usdc": amount_usdc},
        )

    def cashback_balance(self, did: str) -> Dict:
        """Get an agent's USDC cashback balance.

        Args:
            did: Agent DID.

        Returns:
            dict with balance_usdc, pending, and history summary.
        """
        return self._get(HIVEBANK, f"/v1/cashback/balance/{did}")
