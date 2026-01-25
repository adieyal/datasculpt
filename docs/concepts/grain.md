# Grain

The minimal set of columns that uniquely identifies each row.

## What Is Grain?

Grain answers: "What combination of columns has no duplicate values?"

```csv
geo_id,sex,age_group,population
ZA-GP,F,15-24,1200000
ZA-WC,F,15-24,600000
ZA-GP,M,15-24,1150000
```

The grain is `(geo_id, sex, age_group)` — this combination is unique for every row.

## Why Grain Matters

Most data errors are grain errors. They cause silent failures:

| Error | Symptom | Cause |
|-------|---------|-------|
| Duplicated rows after join | Row count explodes | Joined on subset of grain |
| Missing rows after join | Row count drops | Joined on superset of grain |
| Wrong aggregation | Metrics are wrong | Summed at wrong granularity |
| Metric drift | Numbers change over time | Grain changed without notice |

## GrainInference Structure

```python
@dataclass
class GrainInference:
    key_columns: list[str]      # The inferred grain
    confidence: float           # 0.0 to 1.0
    uniqueness_ratio: float     # Fraction of rows that are unique
    evidence: list[str]         # Reasons for the inference
    diagnostics: GrainDiagnostics | None  # Details about issues
```

### Example

```python
>>> result.decision_record.grain
GrainInference(
    key_columns=['geo_id', 'sex', 'age_group'],
    confidence=0.95,
    uniqueness_ratio=1.0,
    evidence=[
        'Combination of geo_id, sex, age_group is unique (8/8 rows)',
        'All columns have low cardinality (dimension-like)',
        'No pseudo-key signals detected'
    ]
)
```

## Uniqueness Ratio

The fraction of rows with unique grain values:

| Ratio | Interpretation |
|-------|----------------|
| 1.0 | Perfect — every row is unique |
| 0.95+ | Good — few duplicates |
| 0.90–0.95 | Acceptable — some duplicates |
| < 0.90 | Problematic — many duplicates |

```python
>>> result.decision_record.grain.uniqueness_ratio
0.95  # 95% of rows are unique
```

## Confidence

How certain Datasculpt is about the grain:

| Confidence | Meaning |
|------------|---------|
| 0.95+ | High confidence — clear grain |
| 0.80–0.95 | Medium — some uncertainty |
| < 0.80 | Low — may need confirmation |

Low confidence triggers questions in interactive mode.

## Diagnostics

When there are issues, diagnostics provide details:

```python
@dataclass
class GrainDiagnostics:
    duplicate_groups: int              # Number of duplicate key combinations
    max_group_size: int                # Largest duplicate group
    rows_with_null_in_key: int         # Rows with nulls in key columns
    null_columns: list[str]            # Which key columns have nulls
    example_duplicate_keys: list[tuple]  # Sample duplicates (max 3)
```

### Example with Issues

```python
>>> result.decision_record.grain.diagnostics
GrainDiagnostics(
    duplicate_groups=3,
    max_group_size=2,
    rows_with_null_in_key=5,
    null_columns=['geo_id'],
    example_duplicate_keys=[
        ('ZA-GP', '2024-01-01'),
        ('ZA-WC', '2024-01-15')
    ]
)
```

## Detection Algorithm

### Step 1: Identify Candidates

Columns that could participate in grain:
- KEY role columns (high uniqueness)
- DIMENSION role columns
- TIME role columns
- INDICATOR_NAME (in long_indicators shape)

Excluded:
- MEASURE columns
- SERIES columns
- METADATA columns

### Step 2: Test Single Columns

Check if any single column is unique:

```python
for col in candidates:
    if df[col].nunique() == len(df):
        # col alone is the grain
        return [col]
```

### Step 3: Apply Pseudo-Key Penalties

Single-column grains are checked for pseudo-key signals:

| Signal | Penalty |
|--------|---------|
| Name matches `row_id`, `row_num` | 0.30 |
| Name matches `index`, `idx` | 0.25 |
| Name matches `uuid`, `guid` | 0.25 |
| Values are monotonic sequence | 0.20 |
| Name matches `created_at`, `timestamp` | 0.20 |

Penalized columns are deprioritized in favor of multi-column grains.

### Step 4: Test Combinations

Test 2-column, then 3-column, then 4-column combinations:

```python
for size in [2, 3, 4]:
    for combo in combinations(candidates, size):
        if df[list(combo)].drop_duplicates().shape[0] == len(df):
            return list(combo)
```

### Step 5: Select Minimal Grain

The first unique combination found is the grain. It's minimal because:
- Single columns are tested first
- Smaller combinations before larger

## Shape-Aware Grain

Grain detection considers the dataset shape:

| Shape | Grain Behavior |
|-------|----------------|
| `long_observations` | Dimensions + time form grain |
| `long_indicators` | Dimensions + time + indicator_name form grain |
| `wide_observations` | Dimensions form grain (measures excluded) |
| `wide_time_columns` | Dimensions form grain (time columns excluded) |
| `series_column` | Dimensions form grain (series column excluded) |

### Example: Long Indicators

```python
# Shape: long_indicators
# Data: geo_id, date, indicator, value

>>> result.decision_record.grain.key_columns
['geo_id', 'date', 'indicator']  # indicator_name is part of grain
```

### Example: Wide Time Columns

```python
# Shape: wide_time_columns
# Data: geo_id, indicator, 2022, 2023, 2024

>>> result.decision_record.grain.key_columns
['geo_id', 'indicator']  # Time columns excluded
```

## Interactive Confirmation

When confidence is low, Datasculpt asks for confirmation:

```python
result = infer("data.csv", interactive=True)

>>> result.pending_questions[0]
Question(
    prompt='Please confirm or select the grain (unique key columns):',
    choices=[
        {'value': ['geo_id', 'date'], 'label': 'Inferred: geo_id, date (95% unique)'},
        {'value': ['geo_id'], 'label': 'geo_id'},
        {'value': ['geo_id', 'date', 'category'], 'label': 'geo_id, date, category'}
    ]
)
```

Apply the user's choice:

```python
from datasculpt import apply_answers

answers = {q.id: ['geo_id', 'date', 'category']}
result = apply_answers(result, answers)

>>> result.decision_record.grain.confidence
1.0  # User confirmed
```

## Configuration

Tune grain detection:

```python
from datasculpt.core.types import InferenceConfig

config = InferenceConfig(
    max_grain_columns=4,            # Max columns to test
    min_uniqueness_confidence=0.95, # Threshold for "good enough"
)
```

## Common Issues

### No Unique Grain Found

If no combination up to `max_grain_columns` is unique:

```python
>>> result.decision_record.grain
GrainInference(
    key_columns=['geo_id', 'date'],  # Best found
    uniqueness_ratio=0.85,           # Only 85% unique
    evidence=['No combination up to 4 columns is fully unique']
)
```

### Nulls in Key Columns

Nulls in key columns reduce uniqueness:

```python
>>> result.decision_record.grain.diagnostics.rows_with_null_in_key
12  # 12 rows have nulls in key columns

>>> result.proposal.warnings
['12 rows have NULL values in key column geo_id']
```

### Pseudo-Key Dominance

If the only unique column is a pseudo-key:

```python
>>> result.decision_record.grain.evidence
['row_id is unique but appears to be a pseudo-key (name pattern)',
 'Falling back to multi-column grain: [geo_id, date]']
```

## See Also

- [Grain Detection Example](../examples/grain-detection.md)
- [Roles](roles.md) — How roles inform grain candidates
- [Decision Records](decision-records.md) — Where grain is stored
