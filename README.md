# AI Support Intelligence Pipeline

> Turn unstructured support case data into a repeatable operational intelligence system.

## The Problem

Support organizations accumulate thousands of cases. None of it is machine-readable
by default, it lives in free-text comments, inconsistent severity classifications,
and agent notes written under pressure. The question "are we getting better at resolving
X?" is nearly impossible to answer at scale without a system like this.

This project is that system.

## What It Does

1. Ingests raw support case data from Salesforce (cases + comment threads)
2. Runs structured AI analysis per case across multiple dimensions
3. Outputs a clean, normalized dataset for BI/analytics consumption
4. Is designed to be repeatable, idempotent, and cost-efficient

### Scoring dimensions per case:

| Dimension | Description |
|-----------|-------------|
| Resolution Quality | Did the agent actually solve the problem? |
| Customer Sentiment | How did the customer feel throughout the interaction? |
| Root Cause Pattern | What category of issue was this really? |
| First Contact Resolution | Resolved in one interaction? |
| Escalation Risk | Would a different agent have escalated this? |
| Documentation Gap | Was this case a symptom of a missing KB article? |

## Architecture

```
┌─────────────────────────────┐
│  Salesforce                 │
│  Cases + Comment Threads    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Data Preparation           │
│  · Deduplication            │
│  · Field normalization      │
│  · Comment thread assembly  │
│  · Case chunking            │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  LLM Analysis Layer         │
│  · Batch processing         │
│  · Rate limit throttling    │
│  · Retry with backoff       │
│  · Structured output        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Output Dataset             │
│  · Idempotent writes        │
│  · Incremental updates      │
│  · Audit trail              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Analytics / Dashboards     │
│  · Quality trend over time  │
│  · Pattern clustering       │
│  · Agent performance        │
│  · Product gap surface      │
└─────────────────────────────┘
```

## Engineering Challenges (and How I Solved Them)

### 1. Context preservation across long case threads

**Problem:** Multi-turn support conversations easily exceed LLM context windows.
Naive chunking loses critical context (e.g., the resolution was mentioned in turn 8
but the problem was introduced in turn 2).

**Solution:** Threads over the token budget are split into overlapping segments
(the last 8 lines of each segment repeat at the start of the next). Each segment is
scored separately and the results are aggregated: sentiment start from the first
segment, sentiment end and root cause from the last, quality and confidence averaged.

### 2. Throttling at scale

**Problem:** Scoring 10,000 cases sequentially would take hours and hit rate limits.
Scoring them in a naive parallel burst triggers rate limit errors instantly.

**Solution:** Token-bucket throttling with per-model rate awareness. Batches are
sized dynamically based on estimated token count, not fixed case counts.

### 3. Idempotent re-runs

**Problem:** If the pipeline crashes at case 3,400, re-running from the start wastes
API credits and time.

**Solution:** Each case gets a content hash. The pipeline skips any case whose hash
matches an already-scored record in the output dataset. Re-runs are safe and cheap.

### 4. Structured output enforcement

**Problem:** LLMs are verbose by default. Downstream SQL queries break when a "score"
field returns `"I would rate this a 7 out of 10"` instead of `7`.

**Solution:** Every response is parsed as JSON and validated for required fields,
types, ranges and allowed enum values. Prose answers are rejected. An invalid response
triggers one re-prompt; if that also fails the case is recorded with an error and is
retried on the next incremental run.

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q
python -m pipeline.run --cases examples/cases.csv --comments examples/comments.csv --out out/ --llm stub
python -m pipeline.run --cases examples/cases.csv --comments examples/comments.csv --out out/ --llm stub --incremental
```

The second run prints `Scored 0 new cases, skipped 30.` because every content hash is already in `out/scores.jsonl`.

Outputs in `out/`: `scores.jsonl` (one score per line), `rca_report.md` and `rca_report.json` (root cause patterns ranked by case count then mean escalation risk, plus documentation gap topics).

Flags: `--llm stub|anthropic` (default `stub`), `--incremental` (skip already scored cases and append; without it `scores.jsonl` is rebuilt). The stub client is deterministic and keyword driven, so no network or key is needed. `--llm anthropic` needs `pip install anthropic` and `ANTHROPIC_API_KEY` in the environment.

The files in `examples/` are synthetic: fictional products, no real people or companies.

## Code Structure

Ingestion from Salesforce is not included; the CLI reads CSVs shaped like the Salesforce `Case` and `CaseComment` objects.

```
ai-support-intelligence/
├── pipeline/
│   ├── prepare.py         # Case + comment join, normalization, hashing, overlapping chunking
│   ├── analyze.py         # Token-bucket throttling, retry with backoff, JSON validation, aggregation
│   ├── llm.py             # LLMClient protocol, StubLLMClient, optional AnthropicClient
│   ├── store.py           # JSONL score store: load_existing_hashes(), append()
│   ├── rca.py             # Group by root cause, rank patterns, documentation gap topics
│   └── run.py             # CLI entry point
├── examples/              # Synthetic cases.csv (30 cases) and comments.csv
├── tests/                 # pytest suite
├── requirements.txt
└── README.md
```

## Using the Layers

`prepare_cases(raw_cases, raw_comments)` takes two pandas DataFrames and returns `PreparedCase` objects. `ThrottledAnalyzer(llm_client, existing_hashes).score_cases(cases)` scores them. `llm_client` is any object exposing `chat(system, user, max_tokens, temperature)`. Pass the content hashes of cases already scored and they are skipped, which makes re-runs idempotent.

## Sample Output

```json
{
  "case_id": "CS-48291",
  "scored_at": "2024-03-15T14:22:00Z",
  "resolution_quality": 8,
  "customer_sentiment_start": "frustrated",
  "customer_sentiment_end": "satisfied",
  "root_cause_pattern": "connector_configuration",
  "first_contact_resolution": false,
  "escalation_risk": "low",
  "documentation_gap": true,
  "documentation_gap_topic": "oauth_token_expiration_handling",
  "confidence": 0.91
}
```

## What I'd Build Next

1. **Real-time scoring**, currently batch; moving to event-driven scoring on case close
2. **Pattern clustering**, use embeddings to cluster similar root causes and auto-generate KB article suggestions
3. **Agent coaching surface**, surface specific scoring breakdowns per agent for quality reviews
4. **Predictive escalation**, train a classifier on historical data to flag high-escalation-risk cases early

## Technologies

`Python` · `pandas` · `SQL` · `Jupyter` · `Salesforce` · `LLM APIs` · `Domo`
