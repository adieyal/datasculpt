# Decision Records

The complete audit trail for every inference.

## What Is a Decision Record?

A decision record captures everything Datasculpt considered when analyzing a dataset:
- What was chosen
- What alternatives were considered
- What evidence supported each choice
- What questions were asked

## Why Decision Records Matter

### Auditability

Every inference can be traced back to evidence:

```
"Why did you call this long_indicators?"
→ Because indicator column has 3 distinct values
→ And value column is numeric with high cardinality
→ And the score was 0.85 vs 0.52 for wide_observations
```

### Reproducibility

Same input + same config = same decision record.

### Debugging

When inference is wrong, the record shows why:

```
"Why didn't you detect the time column?"
→ date_parse_rate was 0.0
→ Because values like "Q1 2024" didn't match the parser
→ → Need to add quarter pattern support
```

### Trust

Users can review decisions before accepting proposals.

## DecisionRecord Structure

```python
@dataclass
class DecisionRecord:
    decision_id: str                    # Unique identifier
    dataset_fingerprint: str            # Hash of input data
    timestamp: datetime                 # When inference ran

    selected_hypothesis: ShapeHypothesis  # Chosen shape
    hypotheses: list[HypothesisScore]   # All shapes with scores

    grain: GrainInference               # Inferred grain

    column_evidence: dict[str, ColumnEvidence]  # Evidence per column

    questions: list[Question]           # Questions generated
    answers: dict[str, Any]             # User answers (if any)
```

## Accessing Decision Records

```python
result = infer("data.csv")
record = result.decision_record

>>> record.decision_id
'dec_a1b2c3d4e5f6'

>>> record.dataset_fingerprint
'sha256_abc123def456...'

>>> record.timestamp
datetime(2024, 1, 15, 10, 30, 45)
```

## Shape Decisions

### Selected Hypothesis

```python
>>> record.selected_hypothesis
<ShapeHypothesis.WIDE_OBSERVATIONS: 'wide_observations'>
```

### Ranked Alternatives

```python
>>> for h in record.hypotheses:
...     print(f"{h.hypothesis.value}: {h.score:.2f}")
...     for reason in h.reasons[:2]:
...         print(f"  - {reason}")

wide_observations: 0.72
  - Multiple numeric columns suggest measures as columns
  - No indicator_name/value column pair detected
long_observations: 0.65
  - Standard observation format
  - Dimensions and measures present
long_indicators: 0.20
  - No indicator column detected
  - No value column paired
```

## Grain Decisions

```python
>>> record.grain
GrainInference(
    key_columns=['geo_id', 'sex', 'age_group'],
    confidence=0.95,
    uniqueness_ratio=1.0,
    evidence=[
        'Combination is unique (8/8 rows)',
        'All columns are dimension-like',
        'No pseudo-key signals'
    ]
)
```

## Column Evidence

Evidence for every column:

```python
>>> record.column_evidence.keys()
dict_keys(['geo_id', 'sex', 'age_group', 'population', 'unemployed'])

>>> ev = record.column_evidence['population']
>>> ev.primitive_type
<PrimitiveType.INTEGER: 'integer'>
>>> ev.distinct_ratio
1.0
>>> ev.role_scores
{<Role.MEASURE>: 0.90, <Role.KEY>: 0.15, ...}
```

## Questions and Answers

### Questions Generated

```python
>>> record.questions
[
    Question(
        id='q_abc123',
        type=<QuestionType.CHOOSE_ONE>,
        prompt='The dataset shape is ambiguous...',
        choices=[...],
        rationale='Score gap is 0.06, below threshold 0.10'
    )
]
```

### User Answers

```python
>>> record.answers
{'q_abc123': 'wide_observations'}  # User selected wide_observations
```

## Persisting Decision Records

Decision records can be serialized for storage:

```python
from datasculpt.decision import serialize_decision_record, deserialize_decision_record

# Serialize to dict (JSON-compatible)
data = serialize_decision_record(record)

# Save to file
import json
with open("decision.json", "w") as f:
    json.dump(data, f, indent=2)

# Load back
with open("decision.json") as f:
    data = json.load(f)
record = deserialize_decision_record(data)
```

## Decision Record Summary

For listing purposes:

```python
@dataclass
class DecisionRecordSummary:
    decision_id: str
    dataset_fingerprint: str
    timestamp: datetime
    path: Path
    selected_hypothesis: str
```

## Linking to Proposals

Each proposal references its decision record:

```python
>>> result.proposal.decision_record_id
'dec_a1b2c3d4e5f6'

# Can retrieve the full record
>>> result.decision_record.decision_id
'dec_a1b2c3d4e5f6'
```

## Reproducibility Check

The fingerprint enables reproducibility verification:

```python
# Run inference
result1 = infer("data.csv")
fingerprint1 = result1.decision_record.dataset_fingerprint

# Run again
result2 = infer("data.csv")
fingerprint2 = result2.decision_record.dataset_fingerprint

>>> fingerprint1 == fingerprint2
True  # Same data → same fingerprint

>>> result1.decision_record.selected_hypothesis == result2.decision_record.selected_hypothesis
True  # Same data → same decision
```

## Debugging with Decision Records

### "Why this shape?"

```python
# Compare top hypotheses
h1, h2 = record.hypotheses[:2]
print(f"{h1.hypothesis.value}: {h1.score:.2f}")
print(f"  Reasons: {h1.reasons}")
print(f"{h2.hypothesis.value}: {h2.score:.2f}")
print(f"  Reasons: {h2.reasons}")
print(f"Gap: {h1.score - h2.score:.2f}")
```

### "Why this role?"

```python
# Look at role scores
ev = record.column_evidence["category"]
sorted_roles = sorted(ev.role_scores.items(), key=lambda x: -x[1])
for role, score in sorted_roles[:3]:
    print(f"{role.value}: {score:.2f}")
```

### "Why this grain?"

```python
# Look at grain evidence
for e in record.grain.evidence:
    print(f"- {e}")

# Check diagnostics
if record.grain.diagnostics:
    print(f"Duplicates: {record.grain.diagnostics.duplicate_groups}")
    print(f"Nulls: {record.grain.diagnostics.rows_with_null_in_key}")
```

## See Also

- [Evidence](evidence.md) — What's captured per column
- [Shapes](shapes.md) — How shapes are scored
- [Grain](grain.md) — How grain is inferred
- [Ambiguous Shape Example](../examples/ambiguous-shape.md) — Questions in action
