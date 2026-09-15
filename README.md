# AI Support Intelligence Pipeline

> Turn unstructured support case data into a repeatable operational intelligence system.

## The Problem

Support organizations accumulate thousands of cases. None of it is machine-readable
by default — it lives in free-text comments, inconsistent severity classifications,
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

**Solution:** Hierarchical chunking — first summarize each comment thread segment,
then pass the summaries + raw final exchange to the scoring call. Preserves context
without exceeding limits.

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

**Solution:** Strict JSON schema enforcement with validation and fallback re-prompting
for any field that doesn't match the expected type.

## Code Structure

```
ai-support-intelligence/
├── pipeline/
│   ├── ingest.py          # Salesforce data extraction
│   ├── prepare.py         # Data preparation + chunking
│   ├── analyze.py         # LLM analysis layer
│   ├── score.py           # Structured scoring + validation
│   └── output.py          # Dataset write + idempotency
├── config/
│   ├── scoring_schema.json  # Output schema definition
│   └── prompts/             # System + user prompt templates
├── tests/
│   ├── test_prepare.py
│   ├── test_analyze.py
│   └── fixtures/            # Sample case data (anonymized)
├── notebooks/
│   └── exploration.ipynb    # EDA + prompt iteration
└── README.md
```

## Running It

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp config/example.env .env
# Fill in: SALESFORCE_URL, LLM_API_KEY, OUTPUT_DATASET_ID

# Run pipeline
python -m pipeline.run --start-date 2024-01-01 --end-date 2024-03-31

# Re-run incrementally (skips already-scored cases)
python -m pipeline.run --incremental
```

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

1. **Real-time scoring** — currently batch; moving to event-driven scoring on case close
2. **Pattern clustering** — use embeddings to cluster similar root causes and auto-generate KB article suggestions
3. **Agent coaching surface** — surface specific scoring breakdowns per agent for quality reviews
4. **Predictive escalation** — train a classifier on historical data to flag high-escalation-risk cases early

## Technologies

`Python` · `pandas` · `SQL` · `Jupyter` · `Salesforce` · `LLM APIs` · `Domo`
