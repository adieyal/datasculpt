# Quickstart

Run your first dataset inference in 5 minutes.

## What You're About to Do

1. Install Datasculpt
2. Run inference on a CSV file
3. Examine the output: shape, grain, and column roles
4. See the decision record explaining each choice

## Install

```bash
pip install datasculpt
```

## Run Inference

Create a sample CSV file:

```csv
geo_id,sex,age_group,population,unemployed,unemployment_rate
ZA-GP,F,15-24,1200000,180000,0.15
ZA-WC,F,15-24,600000,75000,0.125
ZA-GP,M,15-24,1150000,160000,0.139
ZA-WC,M,15-24,580000,70000,0.121
```

Run inference:

```python
from datasculpt import infer

result = infer("demographics.csv")
```

## Examine the Output

### Shape

```python
>>> result.proposal.shape_hypothesis
<ShapeHypothesis.WIDE_OBSERVATIONS: 'wide_observations'>
```

Datasculpt detected this as **wide observations** — a spreadsheet-style format where each row is an observation with measures as columns.

### Grain

```python
>>> result.decision_record.grain.key_columns
['geo_id', 'sex', 'age_group']

>>> result.decision_record.grain.uniqueness_ratio
1.0

>>> result.decision_record.grain.confidence
0.95
```

The grain is the minimal set of columns that uniquely identify each row. Here, the combination of `geo_id`, `sex`, and `age_group` uniquely identifies observations.

### Column Roles

```python
>>> for col in result.proposal.columns:
...     print(f"{col.name}: {col.role.value}")
geo_id: dimension
sex: dimension
age_group: dimension
population: measure
unemployed: measure
unemployment_rate: measure
```

Datasculpt assigned roles based on:
- **Dimensions**: Categorical columns with low cardinality
- **Measures**: Numeric columns with high cardinality

## See the Evidence

Every decision is backed by evidence:

```python
>>> evidence = result.decision_record.column_evidence["population"]
>>> evidence.primitive_type
<PrimitiveType.INTEGER: 'integer'>

>>> evidence.distinct_ratio
1.0

>>> evidence.role_scores
{<Role.MEASURE: 'measure'>: 0.85, <Role.KEY: 'key'>: 0.15, ...}
```

## View Ranked Hypotheses

Datasculpt doesn't just pick a shape — it ranks all candidates:

```python
>>> for h in result.decision_record.hypotheses:
...     print(f"{h.hypothesis.value}: {h.score:.2f}")
wide_observations: 0.72
long_observations: 0.65
long_indicators: 0.20
wide_time_columns: 0.10
series_column: 0.05
```

## Handle Ambiguous Datasets

When Datasculpt isn't confident, it generates questions:

```python
result = infer("ambiguous.csv", interactive=True)

if result.pending_questions:
    for q in result.pending_questions:
        print(q.prompt)
        print(f"  Choices: {[c['label'] for c in q.choices]}")
```

Provide answers to resolve ambiguity:

```python
from datasculpt import apply_answers

answers = {result.pending_questions[0].id: "long_observations"}
result = apply_answers(result, answers)
```

## What Just Happened

Datasculpt ran an 8-stage pipeline:

```
Input → Evidence → Roles → Shape → Grain → Questions → Decision → Proposal
```

1. **Evidence extraction**: Analyzed each column's type, cardinality, null rate, value distribution
2. **Role scoring**: Scored each column against 8 possible roles
3. **Shape detection**: Ranked 5 shape hypotheses
4. **Grain inference**: Found the minimal unique key
5. **Question generation**: Created questions for ambiguous aspects
6. **Decision recording**: Captured the full audit trail
7. **Proposal generation**: Produced output ready for registration

## Next Steps

- [Mental Model](mental-model.md) — Understand the core concepts
- [Examples](../examples/index.md) — See inference on different dataset shapes
- [API Reference](../reference/api.md) — Full function signatures
