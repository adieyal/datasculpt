# Grain Detection

How Datasculpt finds the minimal unique key for a dataset.

## What Is Grain?

The grain is the minimal set of columns that uniquely identifies each row. It's the answer to: "What combination of columns, when taken together, has no duplicates?"

## Why Grain Matters

Most data errors are grain errors:

| Error | Caused By |
|-------|-----------|
| Duplicated rows after join | Wrong grain assumption |
| Aggregation double-counting | Grain finer than expected |
| Missing rows after join | Grain coarser than expected |
| Metric drift | Grain changed over time |

## Basic Detection

```python
from datasculpt import infer

result = infer("demographics.csv")

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

## Detection Algorithm

Datasculpt tests grain candidates in order:

### Step 1: Single Columns

Test each column individually for uniqueness:

```python
# Column cardinality
geo_id: 2 unique values → not unique
sex: 2 unique values → not unique
age_group: 2 unique values → not unique
population: 8 unique values → unique! (but is it a key?)
```

### Step 2: Pseudo-Key Detection

Check if unique columns are "real" keys or just artifacts:

```python
# Pseudo-key signals
population:
  - is_monotonic_sequence: False
  - is_uuid_like: False
  - name_signal_penalty: 0.0  # No "id", "row", "index" in name
  → Probably real key (but measure-like, so demoted)
```

### Step 3: Column Combinations

Test 2-column, then 3-column, then 4-column combinations:

```python
# 2-column tests
(geo_id, sex): 4 unique → not unique for 8 rows
(geo_id, age_group): 4 unique → not unique
(sex, age_group): 4 unique → not unique

# 3-column tests
(geo_id, sex, age_group): 8 unique → unique! ✓
```

### Step 4: Minimality Check

Ensure the key is minimal — no subset is also unique:

```python
# Is (geo_id, sex, age_group) minimal?
(geo_id, sex): 4 unique → not unique (need all 3)
(geo_id, age_group): 4 unique → not unique
(sex, age_group): 4 unique → not unique
→ Yes, all 3 columns are required
```

## Pseudo-Key Penalties

Some columns look unique but aren't meaningful keys:

| Pattern | Penalty | Example |
|---------|---------|---------|
| `row_id`, `row_num` | 0.30 | Auto-generated row numbers |
| `index`, `idx` | 0.25 | DataFrame indices |
| `uuid`, `guid` | 0.25 | Random identifiers |
| `created_at`, `timestamp` | 0.20 | Ingestion timestamps |
| Monotonic integers | 0.20 | 1, 2, 3, 4, 5... |

```python
>>> ev = result.decision_record.column_evidence["row_id"]
>>> ev.role_scores
{<Role.KEY: 'key'>: 0.70, ...}  # Penalized from 1.0
```

## Uniqueness Ratio

When no combination is fully unique, Datasculpt reports the best ratio:

```python
>>> result.decision_record.grain
GrainInference(
    key_columns=['geo_id', 'date'],
    confidence=0.75,
    uniqueness_ratio=0.95,  # 95% of rows are unique
    evidence=[
        'Combination of geo_id, date has 95% unique rows',
        '5% of rows are duplicates'
    ],
    diagnostics=GrainDiagnostics(
        duplicate_groups=3,
        max_group_size=2,
        rows_with_null_in_key=0,
        example_duplicate_keys=[
            ('ZA-GP', '2024-01-01'),
            ('ZA-WC', '2024-01-15'),
            ('ZA-KZN', '2024-02-01')
        ]
    )
)
```

## Handling Nulls in Keys

Nulls in key columns affect uniqueness:

```python
>>> result.decision_record.grain.diagnostics
GrainDiagnostics(
    rows_with_null_in_key=12,
    null_columns=['geo_id'],  # geo_id has nulls
    ...
)

>>> result.proposal.warnings
['12 rows have NULL values in key column geo_id']
```

## Survey-Aware Patterns

Datasculpt recognizes survey and roster patterns:

| Pattern | Interpretation |
|---------|----------------|
| `person_id`, `respondent_id` | Survey respondent |
| `member_id`, `household_member` | Household roster member |
| `visit_id`, `interview_id` | Survey visit/wave |

These get boosted as likely key contributors.

## Interactive Grain Confirmation

When confidence is low, Datasculpt asks for confirmation:

```python
result = infer("data.csv", interactive=True)

>>> result.pending_questions[0]
Question(
    id='q_grain_xyz',
    type=<QuestionType.CHOOSE_MANY>,
    prompt='Please confirm or select the grain (unique key columns) for this dataset:',
    choices=[
        {'value': ['geo_id', 'date'], 'label': 'Inferred: geo_id, date (95% unique)'},
        {'value': ['geo_id'], 'label': 'geo_id'},
        {'value': ['date'], 'label': 'date'},
        {'value': ['geo_id', 'date', 'category'], 'label': 'geo_id, date, category'},
        ...
    ],
    default=['geo_id', 'date'],
    rationale='Uniqueness ratio is 95%, below 100% threshold'
)
```

Provide the correct grain:

```python
from datasculpt import apply_answers

answers = {result.pending_questions[0].id: ['geo_id', 'date', 'category']}
result = apply_answers(result, answers)

>>> result.decision_record.grain.key_columns
['geo_id', 'date', 'category']

>>> result.decision_record.grain.confidence
1.0  # User confirmed
```

## Shape-Aware Grain

Grain detection considers the dataset shape:

| Shape | Grain Behavior |
|-------|----------------|
| `wide_observations` | Dimensions form grain, measures excluded |
| `long_indicators` | Dimensions + indicator_name form grain |
| `wide_time_columns` | Dimensions form grain, time columns excluded |
| `series_column` | Dimensions form grain, series column excluded |

```python
# Long indicators: indicator_name is part of grain
>>> result.decision_record.grain.key_columns
['geo_id', 'date', 'indicator']

# Wide observations: only dimensions
>>> result.decision_record.grain.key_columns
['geo_id', 'sex', 'age_group']
```

## Configuration

Tune grain detection behavior:

```python
from datasculpt.core.types import InferenceConfig

config = InferenceConfig(
    max_grain_columns=4,        # Max columns to test in combination
    min_uniqueness_confidence=0.95,  # Threshold for "unique enough"
)

result = infer("data.csv", config=config)
```

## See Also

- [Wide Observations](wide-observations.md) — Grain from dimensions
- [Long Indicators](long-indicators.md) — Grain includes indicator name
- [Ambiguous Shape](ambiguous-shape.md) — When grain depends on shape interpretation
