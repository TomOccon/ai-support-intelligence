import json
from unittest.mock import patch

import pandas as pd

from pipeline import analyze
from pipeline.analyze import CaseScore, ThrottledAnalyzer
from pipeline.llm import StubLLMClient
from pipeline.prepare import MAX_THREAD_TOKENS, PreparedCase, _chunk_if_needed, _estimate_tokens, prepare_cases
from pipeline.rca import build_rca
from pipeline.run import main
from pipeline.store import ScoreStore

GOOD = {
    "resolution_quality": 7, "customer_sentiment_start": "neutral",
    "customer_sentiment_end": "satisfied", "root_cause_pattern": "sync_delay",
    "first_contact_resolution": True, "escalation_risk": "low",
    "documentation_gap": False, "documentation_gap_topic": None, "confidence": 0.8,
}


def make_case(cid="C1", h="h1", segments=("hello",), tokens=10):
    return PreparedCase(cid, h, "s", "p", "high", "2026-01-01", None, list(segments), tokens)


def make_score(cid, pattern, risk, gap=None, error=None):
    return CaseScore(cid, "h" + cid, "t", 5, "neutral", "neutral", pattern, False, risk,
                     gap is not None, gap, 0.9, 1, error)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def analyzer(client=None, existing=None):
    return ThrottledAnalyzer(client or StubLLMClient(), existing or set())


# chunking
def test_chunking_overlaps_segments():
    text = "\n".join(f"line {i} " + "word " * 200 for i in range(60))
    segs = _chunk_if_needed(text)
    assert len(segs) > 1
    for a, b in zip(segs, segs[1:]):
        assert a.split("\n")[-8:] == b.split("\n")[:8]
    assert "line 59" in segs[-1]


def test_short_thread_not_chunked():
    assert _chunk_if_needed("short text") == ["short text"]


# hashing / idempotency
def _frames():
    cases = pd.DataFrame([{"Id": "A", "Subject": "s", "Description": "oauth token expired", "Priority": "High",
                           "CreatedDate": "2026-01-01", "ClosedDate": None}])
    comments = pd.DataFrame([{"ParentId": "A", "CreatedById": "u", "CreatedDate": "2026-01-02",
                              "Body": "thanks resolved", "IsCustomer__c": "True"}])
    return cases, comments


def test_hash_is_stable_and_content_sensitive():
    cases, comments = _frames()
    h1 = prepare_cases(cases, comments)[0].content_hash
    assert h1 == prepare_cases(cases, comments)[0].content_hash
    comments.loc[0, "Body"] = "something else"
    assert prepare_cases(cases, comments)[0].content_hash != h1


def test_string_false_is_not_customer():
    cases, comments = _frames()
    comments.loc[0, "IsCustomer__c"] = "False"
    assert prepare_cases(cases, comments)[0].thread_segments[0].count("AGENT") == 1


def test_second_run_scores_zero_new(tmp_path, capsys):
    argv = ["--cases", "examples/cases.csv", "--comments", "examples/comments.csv",
            "--out", str(tmp_path), "--llm", "stub", "--incremental"]
    assert main(argv) == 0
    assert "Scored 30 new" in capsys.readouterr().out
    assert main(argv) == 0
    assert "Scored 0 new" in capsys.readouterr().out
    assert len((tmp_path / "scores.jsonl").read_text().splitlines()) == 30
    assert (tmp_path / "rca_report.md").exists()


# validation
def test_prose_answer_rejected():
    assert analyzer()._parse_and_validate("I would rate this a 7 out of 10") is None


def test_valid_json_and_fenced_json_accepted():
    a = analyzer()
    assert a._parse_and_validate(json.dumps(GOOD))["resolution_quality"] == 7
    assert a._parse_and_validate("```json\n" + json.dumps(GOOD) + "\n```") is not None


def test_bad_values_rejected():
    a = analyzer()
    for patch_ in ({"resolution_quality": "seven"}, {"resolution_quality": 11},
                   {"escalation_risk": "extreme"}, {"confidence": 1.5},
                   {"first_contact_resolution": "yes"}):
        assert a._parse_and_validate(json.dumps({**GOOD, **patch_})) is None
    assert a._parse_and_validate("7") is None
    assert a._parse_and_validate(json.dumps({"resolution_quality": 7})) is None


def test_prose_then_json_reprompts():
    client = FakeClient(["I would rate this a 7", json.dumps(GOOD)])
    score = analyzer(client)._score_one(make_case())
    assert client.calls == 2 and score.error is None and score.resolution_quality == 7


def test_all_prose_gives_failed_score():
    score = analyzer(FakeClient(["nope"] * 2))._score_one(make_case())
    assert score.error == "all_segments_failed"


# retry / backoff
def test_retry_backoff_sleeps_exponentially():
    client = FakeClient([RuntimeError("x"), RuntimeError("x"), json.dumps(GOOD)])
    with patch("pipeline.analyze.time.sleep") as sleep:
        out = analyzer(client)._call_llm("t")
    assert out == json.dumps(GOOD)
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2]


def test_retry_gives_up_without_trailing_sleep():
    client = FakeClient([RuntimeError("x")] * 3)
    with patch("pipeline.analyze.time.sleep") as sleep:
        assert analyzer(client)._call_llm("t") is None
    assert sleep.call_count == 2 and client.calls == 3


# throttle
def test_throttle_sleeps_when_budget_exceeded():
    a = analyzer()
    a._budget_per_minute = 100
    with patch("pipeline.analyze.time.sleep") as sleep:
        a._throttle(60)
        assert sleep.call_count == 0
        a._throttle(60)
        assert sleep.call_count == 1
        assert sleep.call_args.args[0] > 0
    assert a._window_tokens == 60


def test_throttle_oversized_single_request_does_not_sleep():
    a = analyzer()
    a._budget_per_minute = 100
    with patch("pipeline.analyze.time.sleep") as sleep:
        a._throttle(500)
    assert sleep.call_count == 0


# aggregation
def test_aggregate_segments():
    a = analyzer()
    s1 = {**GOOD, "resolution_quality": 4, "customer_sentiment_start": "frustrated", "documentation_gap": True,
          "documentation_gap_topic": "topic_a"}
    s2 = {**GOOD, "resolution_quality": 8, "customer_sentiment_end": "angry", "root_cause_pattern": "late"}
    out = a._aggregate_segments(make_case(segments=("a", "b")), [s1, s2])
    assert out.resolution_quality == 6
    assert out.customer_sentiment_start == "frustrated" and out.customer_sentiment_end == "angry"
    assert out.root_cause_pattern == "late" and out.documentation_gap_topic == "topic_a"


def test_prompt_formats_with_json_braces():
    client = FakeClient([json.dumps(GOOD)])
    analyzer(client)._call_llm("body text")  # would raise KeyError with str.format


# rca
def test_rca_ranking_and_gaps():
    scores = [
        make_score("1", "a", "low"), make_score("2", "a", "low"),
        make_score("3", "b", "high", gap="doc_x"), make_score("4", "b", "high", gap="doc_x"),
        make_score("5", "c", "high"),
        make_score("6", "d", "low", error="all_segments_failed"),
    ]
    scores[5].escalation_risk = "unknown"
    r = build_rca(scores)
    assert [p["pattern"] for p in r["patterns"]] == ["b", "a", "c"]  # count desc, then risk desc
    assert r["patterns"][0]["mean_escalation_risk"] == 3
    assert r["documentation_gaps"] == [{"topic": "doc_x", "count": 2}]
    assert r["total_failed"] == 1


def test_store_roundtrip_and_failed_rows_retried(tmp_path):
    st = ScoreStore(tmp_path / "s.jsonl")
    st.append([make_score("1", "a", "low"), make_score("2", "a", "low", error="x")])
    assert st.load_existing_hashes() == {"h1"}
    assert len(st.load_scores()) == 2
