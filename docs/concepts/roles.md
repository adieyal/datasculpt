# Roles

The purpose each column serves in the dataset structure.

## What Is a Role?

A role describes what purpose a column serves — not its data type, but its structural function.

```csv
geo_id,sex,age_group,population,unemployed
ZA-GP,F,15-24,1200000,180000
```

| Column | Type | Role |
|--------|------|------|
| geo_id | string | **dimension** — groups observations |
| sex | string | **dimension** — groups observations |
| age_group | string | **dimension** — groups observations |
| population | integer | **measure** — aggregatable value |
| unemployed | integer | **measure** — aggregatable value |

## The Eight Roles

### Key

Contributes to uniqueness. Part of the grain.

**Signals:**
- High distinct ratio (approaching 1.0)
- No nulls
- Not a pseudo-key (see below)

```python
role_scores[Role.KEY] = 0.95  # Very likely a key
```

### Dimension

Categorical grouping variable. Low cardinality.

**Signals:**
- Low distinct ratio (< 0.1)
- String or categorical type
- Meaningful value labels

```python
role_scores[Role.DIMENSION] = 0.85
```

### Measure

Numeric, aggregatable value. What you SUM, AVG, or COUNT.

**Signals:**
- Numeric type (integer or number)
- High distinct ratio
- Non-negative values (often)

```python
role_scores[Role.MEASURE] = 0.90
```

### Time

Temporal dimension. Dates or timestamps.

**Signals:**
- Parses as date or datetime
- Column name suggests time (date, year, period, month)
- Sequential values

```python
role_scores[Role.TIME] = 0.95
```

### Indicator Name

In unpivoted data, the column that names the metric.

**Signals:**
- Low cardinality
- Values look like concept names (population, gdp, rate)
- Paired with a value column

```python
role_scores[Role.INDICATOR_NAME] = 0.80
```

### Value

In unpivoted data, the column that holds the metric value.

**Signals:**
- Numeric type
- High distinct ratio
- Paired with an indicator name column

```python
role_scores[Role.VALUE] = 0.85
```

### Series

Contains embedded time series (arrays or objects).

**Signals:**
- Structural type is ARRAY or OBJECT
- Arrays have consistent length
- Values are numeric

```python
role_scores[Role.SERIES] = 0.75
```

### Metadata

Descriptive, non-analytical. Notes, comments, labels.

**Signals:**
- High null rate
- Long string values
- Column name suggests metadata (notes, comments, description)

```python
role_scores[Role.METADATA] = 0.60
```

## Role Scoring

Each column is scored against all roles. The highest score wins.

### Scoring Factors

| Factor | Roles Affected |
|--------|----------------|
| Distinct ratio | KEY (high), DIMENSION (low), MEASURE (high) |
| Null rate | KEY (low), METADATA (high) |
| Primitive type | MEASURE (numeric), DIMENSION (string), TIME (date) |
| Column name patterns | All roles (regex matching) |
| Value patterns | INDICATOR_NAME (concept-like), TIME (sequential) |

### Example Scores

```python
>>> ev = result.decision_record.column_evidence["population"]
>>> ev.role_scores
{
    <Role.MEASURE>: 0.90,
    <Role.KEY>: 0.15,
    <Role.DIMENSION>: 0.05,
    <Role.TIME>: 0.0,
    <Role.INDICATOR_NAME>: 0.0,
    <Role.VALUE>: 0.0,
    <Role.SERIES>: 0.0,
    <Role.METADATA>: 0.02
}
```

`population` scores highest as MEASURE because:
- Numeric type (integer)
- High distinct ratio (all unique values)
- Name doesn't match dimension patterns

## Pseudo-Key Detection

Some columns appear unique but aren't meaningful keys:

| Pattern | Example | Penalty |
|---------|---------|---------|
| Row numbers | row_id, row_num | 0.30 |
| Indices | index, idx | 0.25 |
| UUIDs | uuid, guid | 0.25 |
| Timestamps | created_at, ingested_at | 0.20 |
| Monotonic sequences | 1, 2, 3, 4, 5... | 0.20 |

```python
>>> ev.role_scores[Role.KEY]
0.70  # Penalized from 1.0 due to name pattern "row_id"
```

## Shape-Conditional Roles

Role scoring considers the dataset shape:

| Shape | Role Adjustments |
|-------|------------------|
| `long_indicators` | Boost INDICATOR_NAME for low-cardinality string columns |
| `wide_time_columns` | Date-header columns get TIME role |
| `series_column` | Array columns get SERIES role |

```python
# In long_indicators shape:
>>> ev = evidence["indicator"]
>>> ev.role_scores[Role.INDICATOR_NAME]
0.85  # Boosted because shape is long_indicators
```

## Role Assignment

After scoring, roles are assigned:

```python
>>> for col in result.proposal.columns:
...     print(f"{col.name}: {col.role.value}")
geo_id: dimension
sex: dimension
age_group: dimension
population: measure
unemployed: measure
```

## Low-Confidence Assignments

When the top role score is low or close to alternatives, Datasculpt flags uncertainty:

```python
result = infer("data.csv", interactive=True)

>>> result.pending_questions
[
    Question(
        prompt='What is the role of column "category"?',
        choices=[
            {'value': 'dimension', 'label': 'dimension (0.45)'},
            {'value': 'indicator_name', 'label': 'indicator_name (0.42)'},
            {'value': 'metadata', 'label': 'metadata (0.35)'}
        ]
    )
]
```

## Role Implications

### For Aggregation

| Role | Aggregation Behavior |
|------|---------------------|
| MEASURE | Can be SUMmed, AVGed, etc. |
| DIMENSION | Use in GROUP BY |
| TIME | Use in GROUP BY or time series functions |
| INDICATOR_NAME | Filter before aggregating |
| VALUE | Aggregate after filtering by indicator |

### For Joins

| Role | Join Behavior |
|------|---------------|
| KEY | Join on this column |
| DIMENSION | Can join on this (with care) |
| MEASURE | Never join on measures |
| TIME | Join on time ranges |

### For Grain

| Role | Grain Participation |
|------|---------------------|
| KEY | Always part of grain |
| DIMENSION | Often part of grain |
| TIME | Usually part of grain |
| INDICATOR_NAME | Part of grain in long_indicators |
| MEASURE | Never part of grain |
| SERIES | Never part of grain |

## See Also

- [Evidence](evidence.md) — Input to role scoring
- [Grain](grain.md) — How roles inform grain detection
- [Shapes](shapes.md) — Shape-conditional role adjustments
