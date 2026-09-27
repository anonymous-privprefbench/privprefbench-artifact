# PrivPrefBench — Anonymous CODASPY 2027 Artifact

This is the anonymous, offline artifact for the PrivPrefBench Data/Toolset
paper. It contains the benchmark implementation, six synthetic scenarios,
normalized first-turn records, frozen human consent labels, continuation
records, and the analysis scripts used for the reported publication-v3
results.

No API key, provider SDK response payload, author name, affiliation, local
filesystem path, or live credential is required for the commands below. The
included records are synthetic and the included analyses make no network or
paid API calls.

## Reproduce the reported analyses

Python 3.9 or newer is sufficient. The core package has no runtime
dependencies; the test command uses the standard library.

```bash
python3 -m unittest \
  tests.test_publication_human_gold_analysis \
  tests.test_publication_rq3_analysis

python3 scripts/analyze_publication_human_gold.py \
  results/publication-holdout-v3-combined.jsonl \
  annotations/publication-v3/gold.jsonl \
  --output-dir analysis/reproduced-human-gold

python3 scripts/analyze_publication_rq3.py \
  results/publication-holdout-v3-combined.jsonl \
  annotations/publication-v3/gold.jsonl \
  results/publication-v3-rq3-openai.jsonl \
  results/publication-v3-rq3-anthropic.jsonl \
  results/publication-v3-rq3-google.jsonl \
  --output-dir analysis/reproduced-rq3
```

The expected headline results are 1,080 first turns, 64 strict DENY privacy
violations, 477 valid consent requests, 475 ALLOW completions, 476 eligible
continuations, 473 strict-correct continuations, and 476 exact-value-correct
continuations. The test suite checks these values and the reported strata.

The `gold.jsonl` file contains the two-pass human labels. The provider/model
names in the normalized records identify the tested configurations; they do
not imply that a fresh run will be identical to the frozen records.

## Rebuild the paper figure

The prebuilt vector figure is in `paper/figures/codaspy-pvr.pdf`. To regenerate
it from the reproduced first-turn CSV, install Matplotlib and run:

```bash
python3 -m pip install matplotlib
python3 scripts/build_codaspy_figure.py
```

## Scope and limitations

The benchmark uses synthetic profiles and local mock tools. The records do not
represent real bookings, applications, medical appointments, or transmissions
to external services. Results are descriptive for the frozen prompts, model
configurations, and tested provider versions. Re-running live providers is
outside the artifact workflow and would require the corresponding provider
accounts, SDKs, credentials, and a separately approved budget.

## Anonymous review note

This bundle intentionally has no author-identifying metadata. It is intended
to be hosted at an anonymous, reviewer-accessible URL during review. After
acceptance, the authors can replace it with a camera-ready artifact that adds
the permanent project identity and any final documentation.
