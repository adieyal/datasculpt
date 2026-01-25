# Long Indicators

Unpivoted data with indicator name and value columns — common in statistical datasets.

## The Data

```csv
geo_id,date,indicator,value
ZA-GP,2024-01-01,population,1200000
ZA-GP,2024-01-01,unemployed,180000
ZA-GP,2024-01-01,unemployment_rate,0.15
ZA-WC,2024-01-01,population,600000
ZA-WC,2024-01-01,unemployed,75000
ZA-WC,2024-01-01,unemployment_rate,0.125
ZA-GP,2024-02-01,population,1210000
ZA-GP,2024-02-01,unemployed,175000
ZA-GP,2024-02-01,unemployment_rate,0.145
ZA-WC,2024-02-01,population,605000
ZA-WC,2024-02-01,unemployed,72000
ZA-WC,2024-02-01,unemployment_rate,0.119
```

## What It Looks Like

| geo_id | date | indicator | value |
|--------|------|-----------|-------|
| ZA-GP | 2024-01-01 | population | 1200000 |
| ZA-GP | 2024-01-01 | unemployed | 180000 |
| ZA-GP | 2024-01-01 | unemployment_rate | 0.15 |
| ZA-WC | 2024-01-01 | population | 600000 |
| ... | ... | ... | ... |

## The Inference

```python
from datasculpt import infer

result = infer("long_indicators.csv")
```

### Shape Detection

```python
>>> result.proposal.shape_hypothesis
<ShapeHypothesis.LONG_INDICATORS: 'long_indicators'>

>>> result.decision_record.hypotheses[0]
HypothesisScore(
    hypothesis=<ShapeHypothesis.LONG_INDICATORS>,
    score=0.85,
    reasons=[
        'Column "indicator" has low cardinality with concept-like values',
        'Column "value" is numeric with high cardinality',
        'Indicator/value pattern detected'
    ]
)
```

### Grain Detection

```python
>>> result.decision_record.grain
GrainInference(
    key_columns=['geo_id', 'date', 'indicator'],
    confidence=0.92,
    uniqueness_ratio=1.0,
    evidence=[
        'Combination of geo_id, date, indicator is unique',
        'indicator_name column contributes to grain in long_indicators shape'
    ]
)
```

### Role Assignments

| Column | Role | Evidence |
|--------|------|----------|
| geo_id | dimension | Low cardinality, string type |
| date | time | Date type detected, parses as date |
| indicator | indicator_name | Low cardinality, concept-like strings, paired with value column |
| value | value | Numeric, high cardinality, paired with indicator column |

## Why This Shape

Datasculpt detected `long_indicators` because:

1. **Indicator/value pair** — `indicator` column contains concept names (population, unemployed, unemployment_rate), `value` column contains the numeric values
2. **Low cardinality name column** — Only 3 distinct values in `indicator`
3. **High cardinality value column** — Many distinct numeric values
4. **Pattern match** — Column names match common indicator patterns

## Why This Matters

Long indicators format has critical implications for downstream systems:

### Aggregation Safety

In wide format, you can safely compute:
```sql
SELECT geo_id, SUM(population) FROM wide_data GROUP BY geo_id
```

In long format, the equivalent would be **wrong**:
```sql
-- WRONG: Sums population + unemployed + unemployment_rate
SELECT geo_id, SUM(value) FROM long_data GROUP BY geo_id
```

Correct query requires filtering:
```sql
SELECT geo_id, SUM(value) FROM long_data
WHERE indicator = 'population' GROUP BY geo_id
```

### Invariant Integration

When registered as `indicators_long`, Invariant will:
- Block aggregations that don't filter by indicator
- Warn about mixing incompatible indicators
- Validate that indicator names match known concepts

## The Proposal

```python
>>> result.proposal
InvariantProposal(
    dataset_name='long_indicators',
    dataset_kind=<DatasetKind.INDICATORS_LONG>,
    shape_hypothesis=<ShapeHypothesis.LONG_INDICATORS>,
    grain=['geo_id', 'date', 'indicator'],
    columns=[
        ColumnSpec(name='geo_id', role=<Role.DIMENSION>, ...),
        ColumnSpec(name='date', role=<Role.TIME>, ...),
        ColumnSpec(name='indicator', role=<Role.INDICATOR_NAME>, ...),
        ColumnSpec(name='value', role=<Role.VALUE>, ...),
    ],
    warnings=[],
    required_user_confirmations=[]
)
```

## See Also

- [Wide Observations](wide-observations.md) — Same data in wide format
- [Ambiguous Shape](ambiguous-shape.md) — When long vs wide is unclear
