# Series Column

Time series stored as JSON arrays or objects within a single column.

## The Data

```csv
geo_id,indicator,series,frequency,start_date
ZA-GP,gdp_growth,"[10.0, 11.0, 12.0, 11.2, 10.8, 11.5]",monthly,2024-01-01
ZA-WC,gdp_growth,"[7.0, 7.5, 8.0, 7.8, 7.6, 7.9]",monthly,2024-01-01
ZA-GP,inflation,"[5.2, 5.4, 5.3, 5.5, 5.6, 5.4]",monthly,2024-01-01
ZA-WC,inflation,"[4.8, 4.9, 5.0, 5.1, 5.0, 4.9]",monthly,2024-01-01
ZA-KZN,gdp_growth,"[8.5, 8.7, 8.9, 9.0, 8.8, 8.6]",monthly,2024-01-01
ZA-KZN,inflation,"[5.0, 5.1, 5.2, 5.3, 5.2, 5.1]",monthly,2024-01-01
```

## What It Looks Like

| geo_id | indicator | series | frequency | start_date |
|--------|-----------|--------|-----------|------------|
| ZA-GP | gdp_growth | [10.0, 11.0, 12.0, ...] | monthly | 2024-01-01 |
| ZA-WC | gdp_growth | [7.0, 7.5, 8.0, ...] | monthly | 2024-01-01 |
| ZA-GP | inflation | [5.2, 5.4, 5.3, ...] | monthly | 2024-01-01 |
| ... | ... | ... | ... | ... |

## The Inference

```python
from datasculpt import infer

result = infer("series_column.csv")
```

### Shape Detection

```python
>>> result.proposal.shape_hypothesis
<ShapeHypothesis.SERIES_COLUMN: 'series_column'>

>>> result.decision_record.hypotheses[0]
HypothesisScore(
    hypothesis=<ShapeHypothesis.SERIES_COLUMN>,
    score=0.82,
    reasons=[
        'Column "series" contains JSON arrays',
        'Arrays have consistent length across rows',
        'Metadata columns (frequency, start_date) suggest time series context'
    ]
)
```

### Evidence for Array Column

```python
>>> ev = result.decision_record.column_evidence["series"]
>>> ev.structural_type
<StructuralType.ARRAY: 'array'>

>>> ev.parse_results.json_array_rate
1.0

>>> ev.array_profile
ArrayProfile(
    avg_length=6.0,
    min_length=6,
    max_length=6,
    consistent_length=True
)
```

### Grain Detection

```python
>>> result.decision_record.grain
GrainInference(
    key_columns=['geo_id', 'indicator'],
    confidence=0.95,
    uniqueness_ratio=1.0,
    evidence=[
        'Combination of geo_id, indicator is unique',
        'Series column excluded from grain (contains time series data)'
    ]
)
```

### Role Assignments

| Column | Role | Evidence |
|--------|------|----------|
| geo_id | dimension | Low cardinality, string type |
| indicator | dimension | Low cardinality, concept-like values |
| series | series | Array structural type, JSON parses as array |
| frequency | metadata | Describes series properties |
| start_date | time | Date type, series anchor |

## Why This Shape

Datasculpt detected `series_column` because:

1. **Array structural type** — The `series` column contains valid JSON arrays
2. **Consistent array length** — All arrays have 6 elements (max - min <= 1)
3. **Context columns** — `frequency` and `start_date` suggest time series metadata
4. **No time in headers** — Other columns are not date-like

## Array Detection

Datasculpt parses string columns to detect embedded arrays:

```python
# Detected as arrays
"[1, 2, 3, 4, 5]"           # JSON array
"[10.0, 11.0, 12.0]"        # Numeric array
'["a", "b", "c"]'           # String array

# Not detected as arrays
"1, 2, 3, 4, 5"             # Comma-separated (no brackets)
"[invalid json"             # Malformed JSON
```

## Why This Matters

Series column format is compact but requires expansion for analysis.

### Compact Storage

One row = one time series. Efficient for:
- Storage (fewer rows)
- Transfer (smaller payloads)
- APIs (one response = one series)

### Expansion for Analysis

Most tools need the series exploded:

```python
# Compact (current)
# geo_id | indicator | series           | frequency | start_date
# ZA-GP  | gdp_growth| [10.0, 11.0, ...] | monthly   | 2024-01-01

# Expanded (for analysis)
# geo_id | indicator  | date       | value
# ZA-GP  | gdp_growth | 2024-01-01 | 10.0
# ZA-GP  | gdp_growth | 2024-02-01 | 11.0
# ZA-GP  | gdp_growth | 2024-03-01 | 12.0
```

### Metadata Preservation

The `frequency` and `start_date` columns enable correct expansion:
- `frequency=monthly` + `start_date=2024-01-01` → dates increment by month
- `frequency=daily` + `start_date=2024-01-01` → dates increment by day

## The Proposal

```python
>>> result.proposal
InvariantProposal(
    dataset_name='series_column',
    dataset_kind=<DatasetKind.TIMESERIES_SERIES>,
    shape_hypothesis=<ShapeHypothesis.SERIES_COLUMN>,
    grain=['geo_id', 'indicator'],
    columns=[
        ColumnSpec(name='geo_id', role=<Role.DIMENSION>, ...),
        ColumnSpec(name='indicator', role=<Role.DIMENSION>, ...),
        ColumnSpec(name='series', role=<Role.SERIES>,
                   structural_type=<StructuralType.ARRAY>, ...),
        ColumnSpec(name='frequency', role=<Role.METADATA>, ...),
        ColumnSpec(name='start_date', role=<Role.TIME>, ...),
    ],
    warnings=[],
    required_user_confirmations=[]
)
```

## See Also

- [Wide Time Columns](wide-time-columns.md) — Time in headers instead of arrays
- [Long Indicators](long-indicators.md) — Fully expanded format
