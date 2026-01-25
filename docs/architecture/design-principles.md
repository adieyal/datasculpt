# Design Principles

The core principles that guide Datasculpt's design.

## 1. Determinism First

**Given identical input and configuration, Datasculpt produces identical output.**

### What This Means

- No randomness in the inference pipeline
- No LLMs making decisions (advisory only)
- No environment-dependent behavior
- Same file → same fingerprint → same decision

### Why It Matters

- **Reproducibility**: Results can be verified and recreated
- **Testing**: Tests are reliable and don't flake
- **Debugging**: Issues can be reproduced
- **Trust**: Users know what to expect

### Implementation

```python
# Bad: Random sampling
sample = df.sample(1000)  # Different each run

# Good: Deterministic sampling
sample = df.head(1000)  # Same each run
```

```python
# Bad: Order-dependent
for col in df.columns:  # Order may vary
    process(col)

# Good: Sorted order
for col in sorted(df.columns):  # Consistent order
    process(col)
```

## 2. Evidence, Not Authority

**Every inference is scored and justified with evidence.**

### What This Means

- No "because I said so" decisions
- Every choice has a confidence score
- Alternatives are preserved, not discarded
- Users can see why any decision was made

### Why It Matters

- **Auditability**: Trace any decision to evidence
- **Trust**: Users can verify reasoning
- **Debugging**: Understand wrong decisions
- **Override**: Users can correct with knowledge

### Implementation

```python
# Bad: Binary decision
if looks_like_indicator:
    return Role.INDICATOR_NAME

# Good: Scored decision
role_scores = {
    Role.INDICATOR_NAME: 0.85,
    Role.DIMENSION: 0.45,
    Role.METADATA: 0.10,
}
return max(role_scores, key=role_scores.get), role_scores
```

## 3. Shape Before Semantics

**Focus on structure, not meaning.**

### What This Means

- Datasculpt determines layout (long vs wide)
- Datasculpt determines roles (dimension vs measure)
- Datasculpt determines grain (unique key)
- Datasculpt does NOT determine meaning (what "population" means)

### Why It Matters

- **Scope clarity**: Clear boundary with Invariant
- **Universality**: Works without domain knowledge
- **Simplicity**: Fewer assumptions to get wrong

### The Boundary

```
Datasculpt:
  ✓ This is long_indicators shape
  ✓ "indicator" column is the indicator_name
  ✓ "value" column is the value
  ✓ Grain is (geo_id, date, indicator)

Invariant:
  ✓ "population" indicator means total headcount
  ✓ "population" is comparable across years
  ✓ "population" should not be summed across geographies
```

## 4. Minimal Core

**Core functionality requires only pandas.**

### What This Means

- `pip install datasculpt` → works immediately
- Heavy dependencies are optional adapters
- Core is fast and lightweight

### Why It Matters

- **Adoption**: Low barrier to entry
- **Deployment**: Minimal container size
- **Maintenance**: Fewer version conflicts
- **Testing**: Fast test suite

### Implementation

```
datasculpt/
├── core/           # Only pandas
│   ├── evidence.py
│   ├── roles.py
│   └── ...
└── adapters/       # Optional deps
    ├── frictionless_adapter.py  # requires frictionless
    └── dataprofiler_adapter.py  # requires dataprofiler
```

## 5. Multi-Candidate Scoring

**Rank alternatives instead of making binary choices.**

### What This Means

- Shape detection scores all 5 hypotheses
- Role assignment scores all 8 roles
- Grain inference tests multiple candidates
- Ambiguity is surfaced, not hidden

### Why It Matters

- **Visibility**: See what was considered
- **Debugging**: Understand close calls
- **Confidence**: Know when uncertain
- **Interactive**: Present options to users

### Implementation

```python
# Bad: First match wins
for shape in shapes:
    if matches(shape):
        return shape

# Good: Score all, rank
scores = {shape: score_shape(shape) for shape in shapes}
ranked = sorted(scores.items(), key=lambda x: -x[1])
selected = ranked[0][0]
is_ambiguous = ranked[0][1] - ranked[1][1] < threshold
```

## 6. Reversible Decisions

**Users can override any inference.**

### What This Means

- Interactive mode generates questions
- Answers override automated decisions
- Overrides are recorded in decision record

### Why It Matters

- **Control**: Users have final say
- **Domain knowledge**: Humans know context
- **Edge cases**: Handle what automation can't

### Implementation

```python
# Automated decision
result = infer("data.csv")
# shape = wide_observations (automated)

# User override
answers = {question.id: "long_indicators"}
result = apply_answers(result, answers)
# shape = long_indicators (user choice)
# decision_record.answers = {"q_123": "long_indicators"}
```

## 7. Explicit Over Implicit

**Surface assumptions rather than hiding them.**

### What This Means

- Warnings for low confidence
- Required confirmations for risky decisions
- Diagnostics for edge cases
- Notes in evidence explaining signals

### Why It Matters

- **No surprises**: Users know what's uncertain
- **Trust**: System is transparent
- **Debugging**: Issues are visible

### Implementation

```python
# Bad: Silent fallback
if confidence < 0.8:
    return default_grain

# Good: Explicit warning
if confidence < 0.8:
    proposal.warnings.append(
        f"Grain confidence is {confidence:.2f}. "
        "Consider verifying the unique key columns."
    )
    return inferred_grain
```
