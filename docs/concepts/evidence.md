# Evidence

The facts Datasculpt extracts from columns before making inferences.

## What Is Evidence?

Evidence is objective observation — what we see in the data, not what we interpret it to mean.

```python
>>> from datasculpt.core.evidence import extract_dataframe_evidence
>>> evidence = extract_dataframe_evidence(df)
>>> ev = evidence["population"]
>>> ev.primitive_type
<PrimitiveType.INTEGER: 'integer'>
>>> ev.distinct_ratio
1.0
>>> ev.null_rate
0.0
```

Evidence is separated from interpretation so that:
1. Multiple interpretations can use the same evidence
2. Evidence extraction can be tested independently
3. The reasoning chain is visible

## ColumnEvidence Structure

```python
@dataclass
class ColumnEvidence:
    name: str                           # Column name
    primitive_type: PrimitiveType       # string, integer, number, boolean, date, datetime
    structural_type: StructuralType     # scalar, array, object

    # Statistics
    null_rate: float                    # Fraction of nulls (0.0 to 1.0)
    distinct_ratio: float               # Unique values / total rows
    unique_count: int                   # Number of distinct values

    # Value distribution
    value_profile: ValueProfile         # Min, max, mean, ratios

    # Array profile (if structural_type is ARRAY)
    array_profile: ArrayProfile | None

    # Header signals
    header_date_like: bool              # Does the column name look like a date?

    # Parse attempt results
    parse_results: ParseResults         # Date parsing, JSON detection

    # Role likelihoods (populated during role scoring)
    role_scores: dict[Role, float]

    # Notes for debugging
    notes: list[str]
```

## Primitive Types

Datasculpt infers primitive types by examining values:

| Type | Detection |
|------|-----------|
| `string` | Non-numeric text values |
| `integer` | Whole numbers (1, 42, -7) |
| `number` | Floating point (3.14, 0.001) |
| `boolean` | true/false, yes/no, 0/1 |
| `date` | Parseable dates (2024-01-15) |
| `datetime` | Dates with times (2024-01-15T10:30:00) |
| `unknown` | Mixed or unparseable |

```python
>>> ev.primitive_type
<PrimitiveType.INTEGER: 'integer'>
```

## Structural Types

Beyond primitive types, Datasculpt detects structural patterns:

| Type | Detection |
|------|-----------|
| `scalar` | Single values |
| `array` | JSON arrays: `[1, 2, 3]` |
| `object` | JSON objects: `{"a": 1}` |

```python
>>> ev.structural_type
<StructuralType.ARRAY: 'array'>

>>> ev.parse_results.json_array_rate
0.95  # 95% of values parse as JSON arrays
```

## Statistics

### Null Rate

Fraction of values that are null/missing:

```python
>>> ev.null_rate
0.02  # 2% nulls
```

High null rates affect role inference:
- Keys shouldn't have nulls
- Measures with many nulls may be optional
- All-null columns are likely metadata

### Distinct Ratio

Unique values divided by total rows:

```python
>>> ev.distinct_ratio
0.15  # 15% of values are unique
```

| Ratio | Interpretation |
|-------|----------------|
| 1.0 | Every value unique (possible key) |
| 0.8+ | High cardinality (measure-like) |
| 0.1–0.3 | Medium cardinality |
| < 0.1 | Low cardinality (dimension-like) |

### Unique Count

Absolute number of distinct values:

```python
>>> ev.unique_count
12  # 12 distinct values
```

Low unique counts (< 10) suggest categorical/dimension columns.

## Value Profile

Distribution characteristics for numeric columns:

```python
@dataclass
class ValueProfile:
    min_value: float | None
    max_value: float | None
    mean: float | None

    integer_ratio: float        # Values close to integers
    non_negative_ratio: float   # Values >= 0
    bounded_0_1_ratio: float    # Values in [0, 1]
    bounded_0_100_ratio: float  # Values in [0, 100]

    low_cardinality: bool       # unique_count <= 5
    mostly_null: bool           # null_rate > 0.8
```

### Example

```python
>>> ev.value_profile
ValueProfile(
    min_value=0.085,
    max_value=0.15,
    mean=0.113,
    integer_ratio=0.0,
    non_negative_ratio=1.0,
    bounded_0_1_ratio=1.0,      # All values between 0 and 1
    bounded_0_100_ratio=1.0,
    low_cardinality=False,
    mostly_null=False
)
```

This profile suggests a rate or percentage (bounded 0–1).

## Array Profile

For columns with structural type `ARRAY`:

```python
@dataclass
class ArrayProfile:
    avg_length: float
    min_length: int
    max_length: int
    consistent_length: bool     # max - min <= 1
```

### Example

```python
>>> ev.array_profile
ArrayProfile(
    avg_length=6.0,
    min_length=6,
    max_length=6,
    consistent_length=True
)
```

Consistent length arrays suggest time series data.

## Parse Results

Attempts to parse string values:

```python
@dataclass
class ParseResults:
    # Date parsing
    date_parse_rate: float          # Fraction that parse as dates
    has_time: bool                  # Includes time component
    best_date_format: str | None    # Most common format
    date_failure_examples: list[str]  # Values that didn't parse

    # JSON detection
    json_array_rate: float          # Fraction that parse as JSON arrays
```

### Example

```python
>>> ev.parse_results
ParseResults(
    date_parse_rate=0.98,
    has_time=False,
    best_date_format='%Y-%m-%d',
    date_failure_examples=['N/A', 'unknown'],
    json_array_rate=0.0
)
```

## Header Date Detection

Column names are checked for date-like patterns:

```python
>>> ev.header_date_like
True  # Column name is "2024-01"
```

Patterns detected:
- `2024`, `2023` (years)
- `2024-01`, `2024-02` (year-month)
- `2024-Q1`, `2024-Q2` (quarters)
- `Jan 2024`, `February 2024` (month names)

## Role Scores

After role scoring, evidence includes likelihoods for each role:

```python
>>> ev.role_scores
{
    <Role.MEASURE: 'measure'>: 0.85,
    <Role.KEY: 'key'>: 0.10,
    <Role.DIMENSION: 'dimension'>: 0.05,
    <Role.TIME: 'time'>: 0.0,
    ...
}
```

See [Roles](roles.md) for how scores are computed.

## Evidence Extraction

```python
from datasculpt.core.evidence import extract_dataframe_evidence

evidence = extract_dataframe_evidence(df)

for col_name, ev in evidence.items():
    print(f"{col_name}: {ev.primitive_type.value}, {ev.distinct_ratio:.2f} distinct")
```

Evidence extraction:
- Samples up to 1000 rows for large datasets
- Handles mixed types gracefully
- Captures parsing failures for debugging

## See Also

- [Roles](roles.md) — How evidence informs role scores
- [Shapes](shapes.md) — How evidence informs shape detection
- [Decision Records](decision-records.md) — Where evidence is stored
