# Datasculpt

[![CI](https://github.com/adieyal/datasculpt/actions/workflows/ci.yml/badge.svg)](https://github.com/adieyal/datasculpt/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deterministic dataset shape and semantic inference for tabular data.

## The Problem

Before data can be governed, queried, or compared across systems, its structural intent must be understood. Most data systems (catalogs, semantic layers, governance engines) *assume* this understanding exists but don't *produce* it.

## The Solution

Datasculpt infers and explains structural intent:

- **Shape** — Is this long or wide? Time in headers or rows?
- **Grain** — What uniquely identifies each row?
- **Roles** — Which columns are dimensions, measures, or keys?

## What It Is Not

- Not a data catalog (produces metadata, doesn't store it)
- Not an ETL tool (analyzes structure, doesn't transform data)
- Not a semantic layer (understands layout, not meaning)

## Quick Start

```bash
pip install datasculpt
```

```python
from datasculpt import infer

result = infer("data.csv")

print(result.proposal.shape_hypothesis)      # wide_observations
print(result.decision_record.grain.key_columns)  # ['geo_id', 'sex', 'age_group']

for col in result.proposal.columns:
    print(f"{col.name}: {col.role.value}")
# geo_id: dimension
# sex: dimension
# age_group: dimension
# population: measure
# unemployed: measure
```

## Try It

🔬 **[Live Demo](https://adieyal.github.io/datasculpt/demo/)** — Analyze datasets in your browser. No installation, no data leaves your machine.

## Documentation

📚 **[Full Documentation](https://adieyal.github.io/datasculpt/)**

- [Quickstart](https://adieyal.github.io/datasculpt/getting-started/quickstart/) — First inference in 5 minutes
- [Examples](https://adieyal.github.io/datasculpt/examples/) — See inference on different dataset shapes
- [Concepts](https://adieyal.github.io/datasculpt/concepts/) — Understand shapes, roles, and grain
- [API Reference](https://adieyal.github.io/datasculpt/reference/api/) — Function signatures and types

## Key Features

### Five Dataset Shapes

| Shape | Description |
|-------|-------------|
| `long_observations` | Rows are atomic observations |
| `long_indicators` | Unpivoted indicator/value pairs |
| `wide_observations` | Measures as columns |
| `wide_time_columns` | Time periods in column headers |
| `series_column` | Time series as arrays in cells |

### Eight Column Roles

| Role | Purpose |
|------|---------|
| `key` | Contributes to uniqueness |
| `dimension` | Categorical grouping |
| `measure` | Numeric, aggregatable |
| `time` | Temporal dimension |
| `indicator_name` | Names in unpivoted data |
| `value` | Values in unpivoted data |
| `series` | Embedded time series |
| `metadata` | Descriptive, non-analytical |

### Deterministic Inference

Same input → same output. No LLMs, no randomness, no hidden state.

### Evidence-Based

Every decision is scored and justified:

```python
>>> result.decision_record.hypotheses
[
    HypothesisScore(hypothesis=WIDE_OBSERVATIONS, score=0.72, reasons=[...]),
    HypothesisScore(hypothesis=LONG_OBSERVATIONS, score=0.65, reasons=[...]),
]
```

### Interactive Mode

Resolve ambiguity with questions:

```python
result = infer("data.csv", interactive=True)

if result.pending_questions:
    answers = {result.pending_questions[0].id: "long_indicators"}
    result = apply_answers(result, answers)
```

## Installation Options

```bash
# Core only
pip install datasculpt

# With optional adapters
pip install datasculpt[frictionless]   # Schema validation
pip install datasculpt[dataprofiler]   # Statistical profiling
pip install datasculpt[all]            # Everything
```

## Requirements

- Python 3.11+
- pandas 2.0+

## Development

```bash
# Install with dev dependencies
make install-dev

# Run tests
make test

# Lint and format
make lint
make format

# Type checking
make typecheck

# Serve docs locally
make docs-serve
```

## License

MIT
