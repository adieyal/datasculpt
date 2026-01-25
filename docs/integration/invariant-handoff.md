# Invariant Handoff

Connecting Datasculpt to downstream governance.

## The Handoff

Datasculpt produces structural metadata. Invariant consumes it for semantic governance.

```mermaid
flowchart LR
    A[Datasculpt] -->|InvariantProposal| B[Invariant]
```

## InvariantProposal Structure

```python
@dataclass
class InvariantProposal:
    dataset_name: str              # Name for registration
    dataset_kind: DatasetKind      # OBSERVATIONS, INDICATORS_LONG, etc.
    shape_hypothesis: ShapeHypothesis  # Detected shape
    grain: list[str]               # Unique key columns

    columns: list[ColumnSpec]      # Column specifications

    warnings: list[str]            # Issues to review
    required_user_confirmations: list[str]  # Must confirm before registration
    decision_record_id: str        # Link to audit trail
```

## ColumnSpec Structure

```python
@dataclass
class ColumnSpec:
    name: str
    role: Role
    primitive_type: PrimitiveType
    structural_type: StructuralType

    # Hints for Invariant (optional)
    reference_system_hint: str | None   # "ISO-3166-1", "FIPS", etc.
    concept_hint: str | None            # "population", "gdp", etc.
    unit_hint: str | None               # "USD", "persons", etc.
    time_granularity: str | None        # "monthly", "yearly", etc.

    notes: list[str]
```

## DatasetKind Mapping

| Shape | DatasetKind |
|-------|-------------|
| `long_observations` | `OBSERVATIONS` |
| `long_indicators` | `INDICATORS_LONG` |
| `wide_observations` | `OBSERVATIONS` |
| `wide_time_columns` | `TIMESERIES_WIDE` |
| `series_column` | `TIMESERIES_SERIES` |

## Extracting the Proposal

```python
from datasculpt import infer

result = infer("data.csv")
proposal = result.proposal

# For Invariant registration
registration_payload = {
    "name": proposal.dataset_name,
    "kind": proposal.dataset_kind.value,
    "grain": proposal.grain,
    "columns": [
        {
            "name": col.name,
            "role": col.role.value,
            "type": col.primitive_type.value,
            "reference_system": col.reference_system_hint,
            "concept": col.concept_hint,
        }
        for col in proposal.columns
    ],
    "decision_record_id": proposal.decision_record_id,
}
```

## Handling Warnings

Proposals may include warnings that require attention:

```python
if proposal.warnings:
    print("Warnings:")
    for warning in proposal.warnings:
        print(f"  - {warning}")

# Example warnings:
# - Grain confidence is 0.85. Consider verifying the unique key columns.
# - Grain uniqueness is 95%. Dataset may have duplicate rows.
# - Shape detection confidence is low (gap: 0.06). Consider manual review.
```

## Required Confirmations

Some proposals require user confirmation before registration:

```python
if proposal.required_user_confirmations:
    print("Confirmations required:")
    for confirmation in proposal.required_user_confirmations:
        print(f"  - {confirmation}")

    # Block registration until confirmed
    confirmed = get_user_confirmations()
    if not confirmed:
        raise ValueError("Registration blocked: confirmations required")
```

## Linking Decision Records

The proposal links to its decision record for auditability:

```python
# Get the decision record ID
record_id = proposal.decision_record_id

# Store alongside registration
registration_payload["decision_record_id"] = record_id

# Or fetch the full record
from datasculpt.decision import serialize_decision_record

record_data = serialize_decision_record(result.decision_record)
store_decision_record(record_id, record_data)
```

## Enriching Proposals

Add hints for Invariant to use during semantic registration:

```python
# After inference, enrich column specs
for col in result.proposal.columns:
    if col.name == "geo_id":
        col.reference_system_hint = "ISO-3166-2"
    elif col.name == "population":
        col.concept_hint = "population"
        col.unit_hint = "persons"
    elif col.name == "date":
        col.time_granularity = "monthly"
```

## Integration Pattern

Full integration workflow:

```python
from datasculpt import infer, apply_answers
from invariant import register_dataset  # hypothetical

def register_with_invariant(filepath: str):
    # Step 1: Infer structure
    result = infer(filepath, interactive=True)

    # Step 2: Resolve ambiguity
    while result.pending_questions:
        answers = get_user_answers(result.pending_questions)
        result = apply_answers(result, answers)

    # Step 3: Review warnings
    proposal = result.proposal
    if proposal.warnings:
        if not user_acknowledges_warnings(proposal.warnings):
            raise ValueError("Registration aborted: warnings not acknowledged")

    # Step 4: Get confirmations
    if proposal.required_user_confirmations:
        if not user_confirms(proposal.required_user_confirmations):
            raise ValueError("Registration blocked: confirmations required")

    # Step 5: Enrich with domain knowledge
    enrich_proposal_with_domain_hints(proposal)

    # Step 6: Store decision record
    store_decision_record(
        result.decision_record.decision_id,
        serialize_decision_record(result.decision_record)
    )

    # Step 7: Register with Invariant
    register_dataset(
        name=proposal.dataset_name,
        kind=proposal.dataset_kind,
        grain=proposal.grain,
        columns=proposal.columns,
        decision_record_id=proposal.decision_record_id,
    )

    return proposal.dataset_name
```

## Next Steps

- [API Reference](../reference/api.md) — Full function signatures
- [Decision Records](../concepts/decision-records.md) — Audit trail details
