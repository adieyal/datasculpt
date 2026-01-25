# Wide Observations

The most common dataset shape — spreadsheet-style data with measures as columns.

## The Data

```csv
geo_id,sex,age_group,population,unemployed,unemployment_rate
ZA-GP,F,15-24,1200000,180000,0.15
ZA-WC,F,15-24,600000,75000,0.125
ZA-GP,M,15-24,1150000,160000,0.139
ZA-WC,M,15-24,580000,70000,0.121
ZA-GP,F,25-34,1300000,120000,0.092
ZA-WC,F,25-34,650000,55000,0.085
ZA-GP,M,25-34,1250000,110000,0.088
ZA-WC,M,25-34,620000,52000,0.084
```

## What It Looks Like

| geo_id | sex | age_group | population | unemployed | unemployment_rate |
|--------|-----|-----------|------------|------------|-------------------|
| ZA-GP | F | 15-24 | 1200000 | 180000 | 0.15 |
| ZA-WC | F | 15-24 | 600000 | 75000 | 0.125 |
| ... | ... | ... | ... | ... | ... |

## The Inference

```python
from datasculpt import infer

result = infer("wide_observations.csv")
```

### Shape Detection

```python
>>> result.proposal.shape_hypothesis
<ShapeHypothesis.WIDE_OBSERVATIONS: 'wide_observations'>

>>> result.decision_record.hypotheses[0]
HypothesisScore(
    hypothesis=<ShapeHypothesis.WIDE_OBSERVATIONS>,
    score=0.72,
    reasons=[
        'Multiple numeric columns suggest measures as columns',
        'No indicator_name/value column pair detected',
        'No date-like column headers'
    ]
)
```

### Grain Detection

```python
>>> result.decision_record.grain
GrainInference(
    key_columns=['geo_id', 'sex', 'age_group'],
    confidence=0.95,
    uniqueness_ratio=1.0,
    evidence=[
        'Combination of geo_id, sex, age_group is unique',
        'All three columns have low cardinality (dimension-like)',
        'No pseudo-key signals detected'
    ]
)
```

### Role Assignments

| Column | Role | Evidence |
|--------|------|----------|
| geo_id | dimension | Low cardinality (2 values), string type |
| sex | dimension | Low cardinality (2 values), string type |
| age_group | dimension | Low cardinality (2 values), string type |
| population | measure | Integer type, high cardinality, all positive |
| unemployed | measure | Integer type, high cardinality, all positive |
| unemployment_rate | measure | Number type, values in [0, 1] |

## Why This Shape

Datasculpt detected `wide_observations` because:

1. **Multiple numeric columns** — `population`, `unemployed`, `unemployment_rate` are all numeric with high distinct ratios
2. **No indicator pattern** — No column pair matches the indicator_name/value pattern
3. **No time in headers** — Column names don't look like dates
4. **Clear dimension/measure split** — String columns have low cardinality, numeric columns have high cardinality

## The Proposal

```python
>>> result.proposal
InvariantProposal(
    dataset_name='wide_observations',
    dataset_kind=<DatasetKind.OBSERVATIONS>,
    shape_hypothesis=<ShapeHypothesis.WIDE_OBSERVATIONS>,
    grain=['geo_id', 'sex', 'age_group'],
    columns=[
        ColumnSpec(name='geo_id', role=<Role.DIMENSION>, ...),
        ColumnSpec(name='sex', role=<Role.DIMENSION>, ...),
        ColumnSpec(name='age_group', role=<Role.DIMENSION>, ...),
        ColumnSpec(name='population', role=<Role.MEASURE>, ...),
        ColumnSpec(name='unemployed', role=<Role.MEASURE>, ...),
        ColumnSpec(name='unemployment_rate', role=<Role.MEASURE>, ...),
    ],
    warnings=[],
    required_user_confirmations=[]
)
```

## Downstream Implications

When registered with Invariant as `observations`:
- Aggregations over measures are safe
- Joins on grain columns are safe
- Comparing across dimension values is comparable
- The `unemployment_rate` may get flagged as an indicator (derived from other columns)

## See Also

- [Long Indicators](long-indicators.md) — Same data in unpivoted format
- [Grain Detection](grain-detection.md) — How grain is found
