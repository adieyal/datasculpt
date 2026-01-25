# Wide Time Columns

Time series data with time periods encoded in column headers.

## The Data

```csv
geo_id,indicator,2024-01,2024-02,2024-03,2024-04,2024-05,2024-06
ZA-GP,gdp_growth,10.0,11.0,12.0,11.2,10.8,11.5
ZA-WC,gdp_growth,7.0,7.5,8.0,7.8,7.6,7.9
ZA-GP,inflation,5.2,5.4,5.3,5.5,5.6,5.4
ZA-WC,inflation,4.8,4.9,5.0,5.1,5.0,4.9
ZA-KZN,gdp_growth,8.5,8.7,8.9,9.0,8.8,8.6
ZA-KZN,inflation,5.0,5.1,5.2,5.3,5.2,5.1
```

## What It Looks Like

| geo_id | indicator | 2024-01 | 2024-02 | 2024-03 | 2024-04 | 2024-05 | 2024-06 |
|--------|-----------|---------|---------|---------|---------|---------|---------|
| ZA-GP | gdp_growth | 10.0 | 11.0 | 12.0 | 11.2 | 10.8 | 11.5 |
| ZA-WC | gdp_growth | 7.0 | 7.5 | 8.0 | 7.8 | 7.6 | 7.9 |
| ZA-GP | inflation | 5.2 | 5.4 | 5.3 | 5.5 | 5.6 | 5.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |

## The Inference

```python
from datasculpt import infer

result = infer("wide_time_columns.csv")
```

### Shape Detection

```python
>>> result.proposal.shape_hypothesis
<ShapeHypothesis.WIDE_TIME_COLUMNS: 'wide_time_columns'>

>>> result.decision_record.hypotheses[0]
HypothesisScore(
    hypothesis=<ShapeHypothesis.WIDE_TIME_COLUMNS>,
    score=0.88,
    reasons=[
        '6 columns have date-like headers (2024-01, 2024-02, ...)',
        'Header dates form a sequential time series',
        'All date columns contain numeric values'
    ]
)
```

### Evidence for Date Headers

```python
>>> for col in ['2024-01', '2024-02', '2024-03']:
...     ev = result.decision_record.column_evidence[col]
...     print(f"{col}: header_date_like={ev.header_date_like}")
2024-01: header_date_like=True
2024-02: header_date_like=True
2024-03: header_date_like=True
```

### Grain Detection

```python
>>> result.decision_record.grain
GrainInference(
    key_columns=['geo_id', 'indicator'],
    confidence=0.90,
    uniqueness_ratio=1.0,
    evidence=[
        'Combination of geo_id, indicator is unique',
        'Time columns excluded from grain (represent time dimension in headers)'
    ]
)
```

### Role Assignments

| Column | Role | Evidence |
|--------|------|----------|
| geo_id | dimension | Low cardinality, string type |
| indicator | dimension | Low cardinality, concept-like values |
| 2024-01 | time | Header parses as date, numeric values |
| 2024-02 | time | Header parses as date, numeric values |
| ... | time | ... |

## Why This Shape

Datasculpt detected `wide_time_columns` because:

1. **Date-like headers** — Column names `2024-01`, `2024-02`, etc. parse as dates
2. **Multiple time columns** — More than 3 columns with date headers (threshold)
3. **Sequential dates** — The parsed dates form a logical sequence
4. **Numeric values** — All time columns contain numbers

## Header Date Detection

Datasculpt recognizes these header patterns:

| Pattern | Example |
|---------|---------|
| YYYY | 2024, 2023, 2022 |
| YYYY-MM | 2024-01, 2024-02 |
| YYYY-Q# | 2024-Q1, 2024-Q2 |
| Month Year | Jan 2024, February 2024 |
| ISO dates | 2024-01-01 |

## Why This Matters

Wide time columns format requires special handling:

### Reshaping for Analysis

Most analysis tools expect time as a column, not in headers. The data often needs unpivoting:

```python
# Wide (current)
# geo_id | indicator | 2024-01 | 2024-02 | ...

# Long (for analysis)
# geo_id | indicator | date    | value
# ZA-GP  | gdp_growth| 2024-01 | 10.0
# ZA-GP  | gdp_growth| 2024-02 | 11.0
```

### Schema Instability

Each new time period adds a column. This breaks:
- Static column schemas
- Type-safe interfaces
- Caching systems

Datasculpt's detection enables automatic schema evolution handling.

## The Proposal

```python
>>> result.proposal
InvariantProposal(
    dataset_name='wide_time_columns',
    dataset_kind=<DatasetKind.TIMESERIES_WIDE>,
    shape_hypothesis=<ShapeHypothesis.WIDE_TIME_COLUMNS>,
    grain=['geo_id', 'indicator'],
    columns=[
        ColumnSpec(name='geo_id', role=<Role.DIMENSION>, ...),
        ColumnSpec(name='indicator', role=<Role.DIMENSION>, ...),
        ColumnSpec(name='2024-01', role=<Role.TIME>, ...),
        ColumnSpec(name='2024-02', role=<Role.TIME>, ...),
        # ...
    ],
    warnings=[],
    required_user_confirmations=[]
)
```

## See Also

- [Long Indicators](long-indicators.md) — Time in rows, not columns
- [Series Column](series-column.md) — Time series as arrays in cells
