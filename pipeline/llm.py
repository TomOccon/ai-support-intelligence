"""
LLM client abstraction. The analysis layer only needs `chat(...)` returning text.
"""

import json
import os
from typing import Protocol


class LLMClient(Protocol):
    def chat(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        ...


_MARKER = "Case transcript:"

# (keywords, root_cause_pattern, documentation_gap_topic)
_CAUSES = [
    (("oauth", "token", "expired", "reauthorize"), "connector_auth_expiry", "token_refresh_behavior"),
    (("timeout", "slow", "latency", "hangs"), "query_performance", "query_tuning_guidance"),
    (("permission", "access denied", "role", "cannot see"), "permissions_misconfiguration", "role_permission_matrix"),
    (("export", "csv", "download"), "export_formatting", "export_format_options"),
    (("stale", "sync", "not updating", "refresh"), "sync_delay", "sync_schedule_limits"),
]


class StubLLMClient:
    """
    Deterministic, offline client. Reads the transcript out of the user prompt and
    derives a JSON score from keywords. Same input always gives the same output.
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.calls += 1
        transcript = user.split(_MARKER, 1)[-1].lower()

        pattern, topic = "general_how_to", None
        for keywords, name, gap in _CAUSES:
            if any(k in transcript for k in keywords):
                pattern, topic = name, gap
                break

        angry = any(k in transcript for k in ("cancel", "escalate", "legal", "unacceptable"))
        resolved = any(k in transcript for k in ("thanks", "works now", "resolved", "fixed"))
        frustrated_start = any(k in transcript for k in ("urgent", "again", "still", "down", "blocked"))

        if angry:
            end, risk = "angry", "high"
        elif resolved:
            end, risk = "satisfied", "low"
        else:
            end, risk = "frustrated", "medium"

        quality = {"satisfied": 8, "angry": 3, "frustrated": 5}[end]
        return json.dumps({
            "resolution_quality": quality,
            "customer_sentiment_start": "frustrated" if frustrated_start else "neutral",
            "customer_sentiment_end": end,
            "root_cause_pattern": pattern,
            "first_contact_resolution": resolved and transcript.count("[customer]") + transcript.count("customer (") <= 2,
            "escalation_risk": risk,
            "documentation_gap": topic is not None and not resolved,
            "documentation_gap_topic": topic if not resolved else None,
            "confidence": 0.9 if pattern != "general_how_to" else 0.6,
        })


class AnthropicClient:
    """
    Optional real client. The anthropic package is imported lazily so the rest of
    the pipeline runs without it. Reads ANTHROPIC_API_KEY from the environment.
    """

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("Install the anthropic package to use --llm anthropic") from e
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model

    def chat(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        msg = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
