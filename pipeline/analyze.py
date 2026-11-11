"""
LLM analysis layer, scores cases using structured output with throttling,
retry, and idempotency checks.
"""

import datetime
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional
from .prepare import PreparedCase, _estimate_tokens

logger = logging.getLogger(__name__)

# Rate limiting: tokens per minute for the target model
TOKENS_PER_MINUTE = 90_000
SAFETY_FACTOR = 0.85  # stay at 85% of limit


SCORING_SYSTEM_PROMPT = """
You are a support quality analyst. You analyze customer support case transcripts
and score them across several dimensions. You MUST return valid JSON matching
the schema exactly. Do not add explanation outside the JSON object.
"""

SENTIMENT_START = {"frustrated", "neutral", "positive"}
SENTIMENT_END = {"frustrated", "neutral", "satisfied", "angry"}
ESCALATION_LEVELS = {"low", "medium", "high"}
MAX_PARSE_ATTEMPTS = 2  # one re-prompt when the model returns invalid output


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"


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
        # multi-segment: score each segment, aggregate
        segment_scores = []
        for segment in case.thread_segments:
            parsed = None
            for _ in range(MAX_PARSE_ATTEMPTS):
                self._throttle(estimated_tokens=_estimate_tokens(segment))
                raw = self._call_llm(segment)
                parsed = self._parse_and_validate(raw)
                if parsed:
                    break
            if parsed:
                segment_scores.append(parsed)

        if not segment_scores:
            return CaseScore(
                case_id=case.case_id,
                content_hash=case.content_hash,
                scored_at=_now(),
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
        # str.replace, not str.format: the template contains literal JSON braces
        prompt = SCORING_USER_PROMPT.replace("{transcript}", transcript)

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
                if attempt == max_retries - 1:
                    logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Giving up.")
                    break
                wait = 2 ** attempt
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Waiting {wait}s.")
                time.sleep(wait)

        return None

    def _parse_and_validate(self, raw: Optional[str]) -> Optional[dict]:
        if not raw:
            return None

        # extract JSON from response (model may wrap in markdown fences)
        text = raw.strip()
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            text = text.strip()
            if text.lower().startswith("json"):
                text = text[4:]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return None

        if not isinstance(data, dict):
            logger.warning("LLM response is not a JSON object")
            return None

        required_fields = [
            "resolution_quality", "customer_sentiment_start", "customer_sentiment_end",
            "root_cause_pattern", "first_contact_resolution", "escalation_risk",
            "documentation_gap", "documentation_gap_topic", "confidence",
        ]
        if not all(f in data for f in required_fields):
            logger.warning("LLM response missing required fields")
            return None

        try:
            rq = data["resolution_quality"]
            conf = data["confidence"]
            if isinstance(rq, bool) or isinstance(conf, bool):
                raise ValueError("boolean given for numeric field")
            if isinstance(rq, float) and not rq.is_integer():
                raise ValueError("resolution_quality must be an integer")
            rq = int(rq)
            conf = float(conf)
            if not 1 <= rq <= 10:
                raise ValueError("resolution_quality out of range")
            if not 0.0 <= conf <= 1.0:
                raise ValueError("confidence out of range")
            for name in ("first_contact_resolution", "documentation_gap"):
                if not isinstance(data[name], bool):
                    raise ValueError(f"{name} must be a boolean")
            if data["customer_sentiment_start"] not in SENTIMENT_START:
                raise ValueError("bad customer_sentiment_start")
            if data["customer_sentiment_end"] not in SENTIMENT_END:
                raise ValueError("bad customer_sentiment_end")
            if data["escalation_risk"] not in ESCALATION_LEVELS:
                raise ValueError("bad escalation_risk")
            pattern = data["root_cause_pattern"]
            if not isinstance(pattern, str) or not pattern or len(pattern) > 40:
                raise ValueError("bad root_cause_pattern")
            topic = data["documentation_gap_topic"]
            if topic is not None and not isinstance(topic, str):
                raise ValueError("bad documentation_gap_topic")
        except (ValueError, TypeError) as e:
            logger.warning(f"LLM response failed validation: {e}")
            return None

        data["resolution_quality"] = rq
        data["confidence"] = conf
        return data

    def _aggregate_segments(self, case: PreparedCase, scores: list[dict]) -> "CaseScore":
        # for multi-segment cases: use first segment for sentiment_start,
        # last for sentiment_end, average quality/confidence
        return CaseScore(
            case_id=case.case_id,
            content_hash=case.content_hash,
            scored_at=_now(),
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

        if self._window_tokens > 0 and self._window_tokens + estimated_tokens > self._budget_per_minute:
            sleep_for = max(0.0, 60 - elapsed) + 0.5
            logger.info(f"Rate limit headroom reached, sleeping {sleep_for:.1f}s")
            time.sleep(max(0, sleep_for))
            self._window_start = time.monotonic()
            self._window_tokens = 0

        self._window_tokens += estimated_tokens
