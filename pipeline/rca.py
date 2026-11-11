"""
Root-cause analysis over scored cases: group by pattern, rank, surface documentation gaps.
"""

from collections import Counter, defaultdict

from .analyze import CaseScore

RISK_VALUE = {"low": 1, "medium": 2, "high": 3}


def build_rca(scores: list[CaseScore]) -> dict:
    valid = [s for s in scores if not s.error and s.escalation_risk in RISK_VALUE]
    groups: dict[str, list[CaseScore]] = defaultdict(list)
    for s in valid:
        groups[s.root_cause_pattern].append(s)

    patterns = []
    for name, items in groups.items():
        patterns.append({
            "pattern": name,
            "count": len(items),
            "mean_escalation_risk": round(sum(RISK_VALUE[i.escalation_risk] for i in items) / len(items), 2),
            "mean_resolution_quality": round(sum(i.resolution_quality for i in items) / len(items), 2),
            "case_ids": sorted(i.case_id for i in items),
        })
    # highest count first, then highest risk, then name for a stable order
    patterns.sort(key=lambda p: (-p["count"], -p["mean_escalation_risk"], p["pattern"]))

    gap_counts = Counter(s.documentation_gap_topic for s in valid
                         if s.documentation_gap and s.documentation_gap_topic)
    gaps = [{"topic": t, "count": c}
            for t, c in sorted(gap_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    return {
        "total_scored": len(valid),
        "total_failed": len(scores) - len(valid),
        "patterns": patterns,
        "documentation_gaps": gaps,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Root Cause Analysis",
        "",
        f"Cases analyzed: {report['total_scored']}. Failed to score: {report['total_failed']}.",
        "",
        "## Root cause patterns",
        "",
        "| Rank | Pattern | Cases | Mean escalation risk (1-3) | Mean resolution quality |",
        "|------|---------|-------|----------------------------|-------------------------|",
    ]
    for i, p in enumerate(report["patterns"], 1):
        lines.append(f"| {i} | {p['pattern']} | {p['count']} | {p['mean_escalation_risk']} | {p['mean_resolution_quality']} |")
    lines += ["", "## Documentation gap topics", ""]
    if report["documentation_gaps"]:
        lines += [f"- {g['topic']} ({g['count']} cases)" for g in report["documentation_gaps"]]
    else:
        lines.append("None found.")
    return "\n".join(lines) + "\n"
