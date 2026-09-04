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
- **A golden dataset**: 15 hand-labeled example emails in
  [data/golden_dataset.json](data/golden_dataset.json).
- **An eval runner**: runs the classifier over the golden dataset and scores
  category accuracy + a rough summary-quality heuristic, in
  [src/regression_detector/eval_runner.py](src/regression_detector/eval_runner.py).
- **A regression detector**: diffs two eval reports (e.g. prompt v1 vs v2)
  and flags whether the candidate is a regression against the baseline, in
  [src/regression_detector/regression.py](src/regression_detector/regression.py).

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

This prints a pass/fail table per example, aggregate category accuracy, and
an average summary-quality score, then writes a full JSON report to
`reports/eval_<version>_<timestamp>.json`.

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

This prints per-example flips (correct→incorrect and incorrect→correct),
the aggregate accuracy and summary-score deltas, and a verdict. It exits `0`
if no regression is detected and `1` if one is — wire it into CI to fail a
PR automatically. A regression is flagged when:

- category accuracy drops at all (`--accuracy-drop-threshold`, default `0.0`),
- avg summary score drops by more than `0.05` (`--summary-score-drop-threshold`),
- any example that used to be correctly categorized flips to incorrect, or
- any example starts erroring that didn't before.

A full diff report (per-example deltas included) is written to
`reports/diff_<baseline>_vs_<candidate>_<timestamp>.json`.

## What's next

- Wiring the eval + regression check into GitHub Actions to run on every PR
  that touches `prompts/` or the classifier code.
- Slack webhook alerts when `regression.py` detects a regression.
- SQLite storage for historical eval runs + a small dashboard for diffing
  runs over time.
- Docker packaging.
