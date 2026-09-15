"""
Data preparation layer — normalizes raw Salesforce case data and assembles
comment threads before LLM analysis.
"""

import hashlib
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


MAX_THREAD_TOKENS = 3000   # conservative estimate per case
WORDS_PER_TOKEN = 0.75     # rough approximation


@dataclass
class PreparedCase:
    case_id: str
    content_hash: str
    subject: str
    product_area: str
    severity: str
    created_date: str
    closed_date: Optional[str]
    thread_segments: list[str]
    estimated_tokens: int
    metadata: dict = field(default_factory=dict)


def prepare_cases(raw_cases: pd.DataFrame, raw_comments: pd.DataFrame) -> list[PreparedCase]:
    """
    Join cases to their comment threads, normalize fields, chunk if needed.
    Returns a list of PreparedCase objects ready for the analysis layer.
    """
    comment_map = _build_comment_map(raw_comments)
    prepared = []

    for _, row in raw_cases.iterrows():
        case_id = str(row["Id"])
        comments = comment_map.get(case_id, [])
        thread_text = _assemble_thread(row, comments)
        content_hash = _hash_content(thread_text)

        segments = _chunk_if_needed(thread_text)

        prepared.append(PreparedCase(
            case_id=case_id,
            content_hash=content_hash,
            subject=_normalize_field(row.get("Subject", "")),
            product_area=_normalize_field(row.get("Product_Area__c", "unknown")),
            severity=_normalize_severity(row.get("Priority", "")),
            created_date=str(row.get("CreatedDate", "")),
            closed_date=str(row.get("ClosedDate")) if pd.notna(row.get("ClosedDate")) else None,
            thread_segments=segments,
            estimated_tokens=_estimate_tokens(thread_text),
        ))

    return prepared


def _build_comment_map(comments: pd.DataFrame) -> dict[str, list[dict]]:
    """Group comments by ParentId, sorted chronologically."""
    result: dict[str, list[dict]] = {}

    for _, row in comments.sort_values("CreatedDate").iterrows():
        parent = str(row["ParentId"])
        if parent not in result:
            result[parent] = []
        result[parent].append({
            "author": row.get("CreatedById", "unknown"),
            "timestamp": str(row.get("CreatedDate", "")),
            "body": _normalize_field(str(row.get("Body", ""))),
            "is_customer": bool(row.get("IsCustomer__c", False)),
        })

    return result


def _assemble_thread(case_row: pd.Series, comments: list[dict]) -> str:
    """Assemble a readable thread string from case + comments."""
    parts = [
        f"SUBJECT: {case_row.get('Subject', '')}",
        f"DESCRIPTION: {case_row.get('Description', '')}",
        "",
        "--- COMMENT THREAD ---",
    ]

    for i, comment in enumerate(comments):
        speaker = "CUSTOMER" if comment["is_customer"] else "AGENT"
        parts.append(f"[{i+1}] {speaker} ({comment['timestamp']}): {comment['body']}")

    return "\n".join(parts)


def _chunk_if_needed(text: str) -> list[str]:
    """
    If the thread exceeds our token budget, split into overlapping segments.
    Each segment gets ~200 token overlap with the next to preserve context.
    """
    estimated = _estimate_tokens(text)
    if estimated <= MAX_THREAD_TOKENS:
        return [text]

    lines = text.split("\n")
    segments = []
    current_segment_lines = []
    current_tokens = 0
    overlap_lines = []

    for line in lines:
        line_tokens = _estimate_tokens(line)

        if current_tokens + line_tokens > MAX_THREAD_TOKENS and current_segment_lines:
            segments.append("\n".join(current_segment_lines))
            # keep last N lines as overlap context for next segment
            overlap_lines = current_segment_lines[-8:]
            current_segment_lines = overlap_lines.copy()
            current_tokens = _estimate_tokens("\n".join(current_segment_lines))

        current_segment_lines.append(line)
        current_tokens += line_tokens

    if current_segment_lines:
        segments.append("\n".join(current_segment_lines))

    return segments


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _normalize_field(value: str) -> str:
    return " ".join(str(value).split())[:2000]


def _normalize_severity(priority: str) -> str:
    mapping = {
        "Critical": "critical",
        "High": "high",
        "Medium": "medium",
        "Low": "low",
    }
    return mapping.get(priority, "unknown")


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) / WORDS_PER_TOKEN)
