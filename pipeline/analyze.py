"""
LLM analysis layer — scores cases using structured output with throttling,
retry, and idempotency checks.
"""

import json
import time
import logging
from dataclasses import dataclass
from typing import Optional
from .prepare import PreparedCase

logger = logging.getLogger(__name__)

# Rate limiting: tokens per minute for the target model
TOKENS_PER_MINUTE = 90_000
SAFETY_FACTOR = 0.85  # stay at 85% of limit


SCORING_SYSTEM_PROMPT = """
You are a support quality analyst. You analyze customer support case transcripts
and score them across several dimensions. You MUST return valid JSON matching
the schema exactly. Do not add explanation outside the JSON object.
"""

SCORING_USER_PROMPT = """
Analyze this support case and return a JSON object with these exact fields:

{
  "resolution_quality": <integer 1-10>,
  "customer_sentiment_start": <"frustrated" | "neutral" | "positive">,
  "customer_sentiment_end": <"frustrated" | "neutral" | "satisfied" | "angry">,
  "root_cause_pattern": <string, max 40 chars, snake_case>,
  "first_contact_resolution": <boolean>,
  "escalation_risk": <"low" | "medium" | "high">,
  "documentation_gap": <boolean>,
  "documentation_gap_topic": <string or null>,
  "confidence": <float 0.0-1.0>
}

Case transcript:
{transcript}
"""


@dataclass
class CaseScore:
    case_id: str
    content_hash: str
    scored_at: str
    resolution_quality: int
    customer_sentiment_start: str
    customer_sentiment_end: str
    root_cause_pattern: str
    first_contact_resolution: bool
    escalation_risk: str
    documentation_gap: bool
    documentation_gap_topic: Optional[str]
    confidence: float
    segment_count: int
    error: Optional[str] = None


class ThrottledAnalyzer:
    """
    Wraps the LLM client with token-bucket throttling.
    Tracks estimated token usage per minute and sleeps as needed.
    """

    def __init__(self, llm_client, existing_hashes: set[str]):
        self.client = llm_client
        self.existing_hashes = existing_hashes
        self._window_start = time.monotonic()
        self._window_tokens = 0
        self._budget_per_minute = int(TOKENS_PER_MINUTE * SAFETY_FACTOR)

    def score_cases(self, cases: list[PreparedCase]) -> list[CaseScore]:
        scores = []
        skipped = 0

        for case in cases:
            if case.content_hash in self.existing_hashes:
                skipped += 1
                continue

            score = self._score_one(case)
            scores.append(score)

        logger.info(f"Scored {len(scores)} cases, skipped {skipped} (already scored)")
        return scores

    def _score_one(self, case: PreparedCase) -> CaseScore:
        import datetime

        # multi-segment: score each segment, aggregate
        segment_scores = []
        for segment in case.thread_segments:
            self._throttle(estimated_tokens=case.estimated_tokens)
            raw = self._call_llm(segment)
            parsed = self._parse_and_validate(raw)
            if parsed:
                segment_scores.append(parsed)

        if not segment_scores:
            return CaseScore(
                case_id=case.case_id,
                content_hash=case.content_hash,
                scored_at=datetime.datetime.utcnow().isoformat() + "Z",
                resolution_quality=0,
                customer_sentiment_start="unknown",
                customer_sentiment_end="unknown",
                root_cause_pattern="scoring_failed",
                first_contact_resolution=False,
                escalation_risk="unknown",
                documentation_gap=False,
                documentation_gap_topic=None,
                confidence=0.0,
                segment_count=len(case.thread_segments),
                error="all_segments_failed",
            )

        return self._aggregate_segments(case, segment_scores)

    def _call_llm(self, transcript: str, max_retries: int = 3) -> Optional[str]:
        prompt = SCORING_USER_PROMPT.format(transcript=transcript)

        for attempt in range(max_retries):
            try:
                response = self.client.chat(
                    system=SCORING_SYSTEM_PROMPT,
                    user=prompt,
                    max_tokens=512,
                    temperature=0.1,   # low temp = consistent structured output
                )
                return response
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Waiting {wait}s.")
                time.sleep(wait)

        return None

    def _parse_and_validate(self, raw: Optional[str]) -> Optional[dict]:
        if not raw:
            return None

        # extract JSON from response (model may wrap in markdown)
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return None

        required_fields = [
            "resolution_quality", "customer_sentiment_start", "customer_sentiment_end",
            "root_cause_pattern", "first_contact_resolution", "escalation_risk",
            "documentation_gap", "documentation_gap_topic", "confidence",
        ]
        if not all(f in data for f in required_fields):
            logger.warning("LLM response missing required fields")
            return None

        # coerce types
        data["resolution_quality"] = int(data["resolution_quality"])
        data["confidence"] = float(data["confidence"])
        data["first_contact_resolution"] = bool(data["first_contact_resolution"])
        data["documentation_gap"] = bool(data["documentation_gap"])

        return data

    def _aggregate_segments(self, case: PreparedCase, scores: list[dict]) -> "CaseScore":
        import datetime

        # for multi-segment cases: use first segment for sentiment_start,
        # last for sentiment_end, average quality/confidence
        return CaseScore(
            case_id=case.case_id,
            content_hash=case.content_hash,
            scored_at=datetime.datetime.utcnow().isoformat() + "Z",
            resolution_quality=round(sum(s["resolution_quality"] for s in scores) / len(scores)),
            customer_sentiment_start=scores[0]["customer_sentiment_start"],
            customer_sentiment_end=scores[-1]["customer_sentiment_end"],
            root_cause_pattern=scores[-1]["root_cause_pattern"],
            first_contact_resolution=scores[0]["first_contact_resolution"],
            escalation_risk=scores[-1]["escalation_risk"],
            documentation_gap=any(s["documentation_gap"] for s in scores),
            documentation_gap_topic=next(
                (s["documentation_gap_topic"] for s in scores if s.get("documentation_gap_topic")),
                None
            ),
            confidence=sum(s["confidence"] for s in scores) / len(scores),
            segment_count=len(case.thread_segments),
        )

    def _throttle(self, estimated_tokens: int) -> None:
        now = time.monotonic()
        elapsed = now - self._window_start

        if elapsed >= 60:
            self._window_start = now
            self._window_tokens = 0

        if self._window_tokens + estimated_tokens > self._budget_per_minute:
            sleep_for = 60 - elapsed + 0.5
            logger.info(f"Rate limit headroom reached — sleeping {sleep_for:.1f}s")
            time.sleep(max(0, sleep_for))
            self._window_start = time.monotonic()
            self._window_tokens = 0

        self._window_tokens += estimated_tokens
