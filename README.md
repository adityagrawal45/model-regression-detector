# Model Regression Detector

A CI-style pipeline for catching LLM quality regressions before they reach
users.

- **A real LLM feature**: a customer support email classifier (`category` +
  one-sentence `summary`).
- **Versioned prompts**: prompts live as YAML files in [prompts/](prompts/),
  each with a version, timestamp, system prompt, and few-shot examples — this
  is the "code" the eval pipeline runs against.
- **A typed contract**: `PromptConfig` (input) and `ClassificationResult`
  (output), both Pydantic models, in
  [src/regression_detector/config.py](src/regression_detector/config.py).
- **A golden dataset**: 60 hand-written, human-verified test cases (never
  LLM-generated) in [data/golden_dataset.json](data/golden_dataset.json),
  loaded via a typed `GoldenDataset`/`GoldenExample` contract in
  [src/regression_detector/dataset.py](src/regression_detector/dataset.py).
  Each case has a stable id, an `expected_difficulty` (`easy`/`medium`/`hard`),
  and a `notes` field explaining why it's in the dataset. It deliberately
  includes ambiguous emails that straddle two categories, one- or two-word
  emails, typo-heavy emails, sarcastic emails, and mixed-language emails, on
  top of the straightforward cases. The dataset file itself is versioned
  (`version` field) separately from prompt versions, so growing or relabeling
  the eval bar is a visible, trackable change.
- **An async, batched eval runner**: runs every golden-dataset case through
  the classifier concurrently (bounded by `--concurrency`, default 5) via
  `asyncio`, in
  [src/regression_detector/eval_runner.py](src/regression_detector/eval_runner.py).
  Each case is scored on four independent dimensions, stored per-case in the
  report: exact category match (binary), summary relevance (1-5, via an
  LLM-as-judge in
  [src/regression_detector/judge.py](src/regression_detector/judge.py)),
  latency (ms), and token usage (prompt/completion/total).
- **A regression detector**: diffs two eval reports (e.g. prompt v1 vs v2)
  and flags whether the candidate is a regression against the baseline, in
  [src/regression_detector/regression.py](src/regression_detector/regression.py).
  Beyond the overall pass-rate delta, it breaks accuracy deltas down
  per-category and classifies each delta's magnitude as noise, a warning, or
  critical (default 3% / 8%, both configurable) — so a couple of flipped
  cases in an otherwise-healthy category doesn't get lost in the aggregate
  number, and small run-to-run noise doesn't cry wolf.

LLM provider is [Groq](https://groq.com) (OpenAI-compatible API), but the
classifier and eval runner only depend on the `LLMClient` protocol in
[llm_client.py](src/regression_detector/llm_client.py) — swapping providers
later is a one-line change. A `MockClient` is included so tests and demo runs
work fully offline with no API key.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then fill in GROQ_API_KEY
```

Get a Groq API key at https://console.groq.com/keys.

## Running the eval

With a real key (loads `.env` automatically if you `pip install python-dotenv`
and load it, or just set the env var directly):

```bash
$env:GROQ_API_KEY = "gsk_..."
python -m regression_detector.eval_runner --prompt prompts/classifier_v1.yaml --dataset data/golden_dataset.json
```

Without a key (offline / demo mode, uses `MockClient` automatically):

```bash
python -m regression_detector.eval_runner --mock
```

This prints a pass/fail table per example (category match, heuristic and
LLM-judge summary scores, latency), aggregate category accuracy, average
summary scores, average latency, total tokens used, and accuracy breakdowns
by difficulty (`easy`/`medium`/`hard`) and by category, then writes a full
JSON report — with all four scoring dimensions stored per case — to
`reports/eval_<version>_<timestamp>.json`. Pass `--concurrency N` to change
how many cases run at once (default 5); each case makes two LLM calls (one to
classify, one for the judge to score the summary), both batched.

Note: `MockClient`'s keyword matching is a thin offline stand-in, not a real
classifier it won't ace the harder (ambiguous/sarcastic/mixed-language)
cases in the golden dataset, and that's expected. Those cases exist to
stress a *real* model; a real Groq run is expected to score meaningfully
higher on them than `MockClient` does.

## Running tests

```bash
pytest
```

All tests run against `MockClient` — no network or API key required.

## Iterating on prompts

To test a new prompt version: copy `prompts/classifier_v1.yaml` to
`prompts/classifier_v2.yaml`, edit the system prompt / few-shot examples, bump
`version`, and re-run the eval with `--prompt prompts/classifier_v2.yaml`.
Each run writes a timestamped report to `reports/`.

## Detecting regressions

Compare a baseline report against a candidate report to see whether a prompt
or model change made things worse:

```bash
python -m regression_detector.regression \
    --baseline reports/eval_v1_<timestamp>.json \
    --candidate reports/eval_v2_<timestamp>.json
```

This prints per-example flips (correct→incorrect and incorrect→correct), the
overall pass-rate and summary-score deltas, a per-category accuracy delta
table, and a verdict. It exits `0` if no regression is detected and `1` if
one is — wire it into CI to fail a PR automatically. A regression is flagged
(`is_regression`) when:

- category accuracy drops at all (`--accuracy-drop-threshold`, default `0.0`),
- avg summary score drops by more than `0.05` (`--summary-score-drop-threshold`),
- any example that used to be correctly categorized flips to incorrect,
- any example starts erroring that didn't before, or
- the overall or any per-category accuracy delta crosses the warning/critical
  severity thresholds below.

Separately from the pass/fail gate above, every accuracy delta (overall and
per-category) is classified on a statistical-significance-style severity
scale, since on a modest golden dataset a couple of flipped cases can be
noise or a real problem depending on which category they land in:

- **none**: drop is below `--warning-delta-threshold` (default `3%`)
- **warning**: drop is at or above the warning threshold but below `--critical-delta-threshold` (default `8%`)
- **critical**: drop is at or above the critical threshold

Both thresholds are tunable per run — a 20-case dataset needs a looser bar
than a 500-case one. A full diff report (per-example deltas, per-category
deltas and severities, and both threshold values used, all included) is
written to `reports/diff_<baseline>_vs_<candidate>_<timestamp>.json`.

## HTML diff report

Every `regression.py` run also writes a self-contained HTML report (inline
CSS + an inline SVG chart, no external assets) next to the JSON diff report —
`reports/diff_<baseline>_vs_<candidate>_<timestamp>.html` — built by
[src/regression_detector/report_html.py](src/regression_detector/report_html.py).
It has: run metadata (prompt version, model, timestamp) for both runs; a
scorecard comparing every scoring dimension (category accuracy, both summary
scores, latency, tokens) candidate vs baseline; a side-by-side table of every
regressed case (old category/summary vs new); and a trend chart of category
accuracy over the last N runs, with the drift rolling average overlaid when
available. Point `--history-dir` at the directory of past `eval_*.json`
reports to populate the trend chart (defaults to the baseline's directory).
Pass `--no-html` to skip it, or `--html-out <path>` to control where it goes.

## Slack alerts

Pass `--slack` to `regression.py` to post a Block Kit message to a Slack
incoming webhook — set `SLACK_WEBHOOK_URL` (or pass `--slack-webhook-url`)
and, to include a link to the HTML report, `--report-url <public URL>` (the
local file path is used as a fallback if omitted, which is only useful if
that path is itself reachable, e.g. a shared CI artifacts URL). The message
carries a PASS/WARN/FAIL status, the headline numbers ("N regressions
detected, accuracy dropped from X% to Y%"), any per-category severities, a
drift line if `detect_drift` flagged one, and the report link. See
[src/regression_detector/alerts.py](src/regression_detector/alerts.py) — the
actual HTTP POST is a single injectable function, so `build_slack_message`
can be tested without ever hitting the network.

## Drift detection

Per-run diffs catch a single bad prompt/model change; they don't catch a
*slow* decline where each run only drops a point or two — well under the
per-run warning threshold — but the trend adds up over many runs.
[src/regression_detector/drift.py](src/regression_detector/drift.py) tracks a
rolling average of category accuracy across a history of eval runs
(`--drift-window`, default 7) and compares the current window's average
against the window before it, using the same warning/critical severity scale
as `regression.py`. A `--drift-absolute-floor` is also available for a fixed
"never go below this" bar, independent of trend. `regression.py --slack`
folds a "slow drift" line into the Slack alert automatically when drift is
detected, even on an otherwise-passing run.

## What's next

- Wiring the eval + regression check into GitHub Actions to run on every PR
  that touches `prompts/` or the classifier code.
- SQLite storage for historical eval runs + a small dashboard for diffing
  runs over time (currently reports/ + directory globbing).
- Docker packaging.
