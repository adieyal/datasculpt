# Datasculpt Design Document

**Deterministic Dataset Shape & Semantic Inference for Invariant**

Version: 0.1.0
Status: Implementation-Ready

---

## 1. Executive Summary

Datasculpt is a standalone Python library that infers the structural intent of tabular datasets before registration with Invariant. It transforms opaque CSV/spreadsheet data into structured, evidence-backed proposals that describe:

- **Shape**: Is this long-form observations, wide-form, time-in-columns, or series objects?
- **Column Roles**: Which columns are keys, dimensions, measures, time indices, or series data?
- **Grain**: What is the minimal set of columns that uniquely identifies each observation?

Datasculpt produces **hypotheses, not decisions**. Every inference is explainable, auditable, and reversible.

---

## 2. Problem Statement

### 2.1 The Challenge

Real-world tabular data arrives with:
- Arbitrary column names with no semantic meaning
- Inconsistent formatting (dates as strings, numbers as text)
- No explicit schema or metadata
- Ambiguous structure (is this wide or long format?)

Invariant requires structural clarity to reason about dataset compatibility, aggregation correctness, and semantic equivalence. Without understanding structure, semantic reasoning fails.

### 2.2 What Exists Today

| Tool | Purpose | Gap |
|------|---------|-----|
| Frictionless | Schema inference | Types only, no roles/shape/grain |
| DataProfiler | Statistical profiling | Heavy, no structural inference |
| pandas | Data manipulation | No inference at all |

**Datasculpt fills the gap**: structured inference that existing tools don't provide.

---

## 3. Design Principles

### 3.1 Determinism First

Given identical input and configuration, Datasculpt produces identical output.

- No hidden randomness
- No LLMs in the decision loop (advisory only)
- Reproducible across runs

### 3.2 Evidence, Not Authority

Datasculpt never asserts truth. Every inference is:

- **Scored**: Confidence level from 0.0 to 1.0
- **Justified**: Reasons for and against each hypothesis
- **Reversible**: User can override any decision

Ambiguity is surfaced explicitly, never hidden.

### 3.3 Shape Before Semantics

Datasculpt focuses on **structure**:
- Layout (wide vs long)
- Roles (key vs measure)
- Grain (uniqueness)

**Semantic identity** (what a variable *means*) belongs to Invariant's downstream layers. Datasculpt's job ends once structural intent is clear.

### 3.4 Minimal Core Dependencies

Core inference relies only on:
- `pandas` (data manipulation)
- Python standard library

Heavy profilers are **optional plugins**, never required for core functionality.

---

## 4. Architecture

### 4.1 High-Level Flow

```
┌─────────────────┐
│  Tabular Input  │  (CSV, DataFrame)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   Evidence Extraction       │  Deterministic column analysis
│   (per-column statistics)   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Role Scoring              │  Score each column for each role
│   (key/dim/measure/time)    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Shape Hypothesis Scoring  │  Score competing dataset shapes
│   (long/wide/series/etc)    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Grain Inference           │  Find minimal unique key set
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Decision Record           │  All hypotheses + evidence
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Invariant Proposal        │  Ready for registration
└─────────────────────────────┘
```

### 4.2 Package Structure

```
src/datasculpt/
├── __init__.py              # Public API exports
├── py.typed                 # PEP 561 marker
├── core/
│   ├── __init__.py
│   ├── evidence.py          # ColumnEvidence extraction
│   ├── roles.py             # Role scoring logic
│   ├── shapes.py            # Shape hypothesis scoring
│   ├── grain.py             # Grain/key inference
│   └── types.py             # Core type definitions
├── pipeline.py              # Main inference pipeline
├── proposal.py              # InvariantProposal generation
├── decision.py              # DecisionRecord management
├── config.py                # Configuration and thresholds
└── adapters/                # Optional profiler adapters
    ├── __init__.py
    ├── frictionless.py      # Frictionless integration
    ├── dataprofiler.py      # DataProfiler integration
    └── reporting.py         # ydata-profiling reports
```

### 4.3 Dependency Tiers

| Tier | Package | Dependencies | Purpose |
|------|---------|--------------|---------|
| Core | `datasculpt` | pandas only | All inference logic |
| Optional | `datasculpt[frictionless]` | + frictionless | Schema hints |
| Optional | `datasculpt[dataprofiler]` | + dataprofiler | Richer statistics |
| Optional | `datasculpt[reporting]` | + ydata-profiling | HTML reports |
| Dev | `datasculpt[dev]` | + pytest, ruff, mypy | Development tools |

---

## 5. Core Concepts

### 5.1 ColumnEvidence

A normalized representation of everything known about a column:

```python
@dataclass
class ColumnEvidence:
    name: str
    primitive_type: PrimitiveType      # string, integer, number, date, etc.
    structural_type: StructuralType    # scalar, array, object

    # Statistics
    null_rate: float                   # 0.0 to 1.0
    distinct_ratio: float              # unique/total

    # Parse attempts
    parse_results: dict[str, float]    # {"date": 0.95, "json_array": 0.0, ...}

    # Role likelihoods (computed)
    role_scores: dict[Role, float]     # {"key": 0.8, "measure": 0.1, ...}

    # External profiler data (optional)
    external: dict[str, Any]           # {"frictionless": {...}, ...}

    # Explanatory notes
    notes: list[str]
```

This abstraction isolates the system from profiler-specific schemas.

### 5.2 Roles

Columns are scored for each role:

| Role | Description | Signals |
|------|-------------|---------|
| `key` | Primary identifier | High cardinality, low nulls |
| `dimension` | Categorical grouping | Low-medium cardinality, strings |
| `measure` | Numeric observation | Numeric type, varying values |
| `time` | Temporal index | Date/datetime parsing success |
| `indicator_name` | Variable name in long format | Low cardinality, strings |
| `value` | Value in long format | Numeric, paired with indicator |
| `series` | Array/object containing time series | JSON array parsing success |
| `metadata` | Ancillary information | Catch-all |

### 5.3 Shape Hypotheses

Datasculpt evaluates competing hypotheses about dataset structure:

| Hypothesis | Description | Example |
|------------|-------------|---------|
| `long_observations` | Rows are atomic observations with dimensions + measures | Standard tidy data |
| `long_indicators` | Has indicator_name/value columns | Unpivoted data |
| `wide_observations` | Multiple measures as columns | Spreadsheet-style |
| `wide_time_columns` | Time periods encoded in column headers | "2024-01", "2024-02", ... |
| `series_column` | Time series stored as arrays/objects | JSON arrays in cells |

Each hypothesis is scored based on column evidence. The highest-scoring hypothesis is selected, but all scores are recorded.

### 5.4 Grain

The grain is the **minimal set of columns that uniquely identifies an observation**.

Inference approach:
1. Identify candidate key columns (high cardinality, low nulls)
2. Test column combinations for uniqueness
3. Prefer smaller key sets
4. Report confidence based on uniqueness ratio

```python
@dataclass
class GrainInference:
    key_columns: list[str]
    confidence: float              # 0.0 to 1.0
    uniqueness_ratio: float        # 1.0 = perfectly unique
    evidence: list[str]            # Justification
```

### 5.5 DecisionRecord

Every inference run produces a complete audit trail:

```python
@dataclass
class DecisionRecord:
    decision_id: str
    dataset_fingerprint: str
    timestamp: datetime

    selected_hypothesis: str
    hypotheses: list[HypothesisScore]    # All hypotheses with scores

    grain: GrainInference
    column_evidence: dict[str, ColumnEvidence]

    questions: list[Question]            # Unresolved ambiguities
    answers: dict[str, Any]              # User/LLM responses
```

### 5.6 InvariantProposal

The final output, ready for Invariant registration:

```python
@dataclass
class InvariantProposal:
    dataset_name: str
    dataset_kind: DatasetKind
    shape_hypothesis: str
    grain: list[str]
    columns: list[ColumnSpec]

    warnings: list[str]
    required_user_confirmations: list[str]
    decision_record_id: str
```

---

## 6. Inference Logic

### 6.1 Evidence Extraction

For each column, compute:

1. **Primitive type detection**
   - Attempt parsing: integer, float, date, datetime, boolean
   - Fall back to string

2. **Structural type detection**
   - Check for JSON arrays/objects in string columns
   - Detect nested structures

3. **Statistical properties**
   - Null rate
   - Distinct value ratio
   - Value distribution characteristics

4. **Header analysis**
   - Check if column name looks like a date (for wide_time_columns)
   - Pattern matching on common naming conventions

### 6.2 Role Scoring

Each column receives a score (0.0 to 1.0) for each role:

```python
def score_key(evidence: ColumnEvidence) -> float:
    score = 0.0
    if evidence.distinct_ratio > 0.9:
        score += 0.4
    if evidence.null_rate < 0.01:
        score += 0.3
    if evidence.primitive_type in (PrimitiveType.STRING, PrimitiveType.INTEGER):
        score += 0.2
    if "_id" in evidence.name.lower() or "code" in evidence.name.lower():
        score += 0.1
    return min(score, 1.0)
```

Similar scoring functions for each role.

### 6.3 Shape Hypothesis Scoring

Dataset-level scoring based on column evidence:

```python
def score_wide_time_columns(columns: list[ColumnEvidence]) -> float:
    date_headers = sum(1 for c in columns if looks_like_date(c.name))
    if date_headers >= 3:
        return 0.8 + (0.2 * (date_headers / len(columns)))
    return 0.0
```

### 6.4 Grain Inference

```python
def infer_grain(df: pd.DataFrame, evidence: dict[str, ColumnEvidence]) -> GrainInference:
    # 1. Rank columns by key likelihood
    candidates = sorted(
        evidence.values(),
        key=lambda e: e.role_scores.get(Role.KEY, 0),
        reverse=True
    )

    # 2. Try single columns first
    for col in candidates[:5]:
        if df[col.name].is_unique:
            return GrainInference(
                key_columns=[col.name],
                confidence=0.95,
                uniqueness_ratio=1.0,
                evidence=["Single column provides uniqueness"]
            )

    # 3. Try combinations
    # ... (combinatorial search with early exit)
```

---

## 7. Interactive Inference

When ambiguity remains after deterministic analysis:

1. **Generate questions**
   ```python
   Question(
       id="shape_ambiguity",
       type="choose_one",
       prompt="The dataset could be interpreted as wide_observations or long_indicators. Which is correct?",
       choices=[
           {"value": "wide_observations", "label": "Wide: each column is a measure"},
           {"value": "long_indicators", "label": "Long: indicator/value pattern"},
       ],
       rationale="Confidence difference between hypotheses is <10%"
   )
   ```

2. **Accept answers**
   - From user interaction
   - From LLM assistant (advisory)

3. **Re-run pipeline with constraints**
   - Locked decisions are respected
   - Other inferences may change

LLMs are advisory only; deterministic logic remains authoritative.

---

## 8. Configuration

All thresholds are configurable via `InferenceConfig`:

```python
@dataclass
class InferenceConfig:
    # Role scoring thresholds
    key_cardinality_threshold: float = 0.9
    dimension_cardinality_max: float = 0.1

    # Shape detection
    min_time_columns_for_wide: int = 3

    # Grain inference
    max_grain_columns: int = 4
    min_uniqueness_confidence: float = 0.95

    # Ambiguity thresholds
    hypothesis_confidence_gap: float = 0.1  # Require this gap to avoid questions

    # Optional adapters
    use_frictionless: bool = False
    use_dataprofiler: bool = False
```

Defaults are conservative and explicit.

---

## 9. API Design

### 9.1 Primary Interface

```python
from datasculpt import infer, InferenceConfig

# Simple usage
proposal = infer("data.csv")

# With configuration
config = InferenceConfig(use_frictionless=True)
proposal = infer(df, config=config)

# Access decision record
record = proposal.get_decision_record()

# Interactive mode
proposal = infer("data.csv", interactive=True)
for question in proposal.pending_questions:
    answer = get_user_input(question)
    proposal = proposal.answer(question.id, answer)
```

### 9.2 Programmatic Access

```python
from datasculpt.core import extract_evidence, score_roles, score_shapes
from datasculpt.core.grain import infer_grain

# Step-by-step for custom pipelines
evidence = extract_evidence(df)
roles = score_roles(evidence)
shapes = score_shapes(evidence, roles)
grain = infer_grain(df, evidence)
```

---

## 10. Testing Strategy

### 10.1 Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit | Individual function correctness | `tests/unit/` |
| Integration | Full pipeline on fixtures | `tests/integration/` |
| Contract | Proposal schema validity | `tests/contract/` |
| Property | Invariants (order, naming) | `tests/property/` |

### 10.2 Canonical Fixtures

```
tests/fixtures/
├── wide_observations.csv      # Standard wide format
├── long_indicators.csv        # Unpivoted format
├── wide_time_columns.csv      # Time in headers
├── series_column.csv          # JSON arrays
├── ambiguous_shape.csv        # Requires disambiguation
└── missing_grain.csv          # No clear unique key
```

### 10.3 Key Test Properties

- **Determinism**: Same input → same output
- **Order invariance**: Column/row order doesn't affect inference
- **Naming robustness**: Arbitrary column names handled
- **Schema validity**: All proposals validate against JSON schema

---

## 11. Integration with Invariant

Datasculpt outputs an **InvariantProposal**, not direct writes.

Invariant is responsible for:
- Catalog persistence
- Semantic validation
- Reference system resolution
- Compatibility checks

Datasculpt provides the structural scaffolding that enables Invariant's semantic reasoning.

```
┌──────────────┐        ┌─────────────────┐        ┌──────────────┐
│  Raw Data    │───────▶│   Datasculpt    │───────▶│   Invariant  │
│  (CSV, etc)  │        │   (Structure)   │        │  (Semantics) │
└──────────────┘        └─────────────────┘        └──────────────┘
```

---

## 12. Success Criteria

Datasculpt is successful if:

1. **Most datasets register without hand-written schemas**
   - 80%+ of common tabular formats inferred correctly

2. **Assumptions are visible and reviewable**
   - Every inference has a justification
   - Users can trace decisions back to evidence

3. **Invariant compatibility errors decrease**
   - Better upfront structure → fewer downstream errors

4. **Users trust the system**
   - Because it explains itself, not because it's "magic"

---

## 13. Open Questions & Future Work

### 13.1 Deferred Decisions

- Support for nested/object columns beyond arrays
- Incremental inference for streaming data
- Integration with recipe/budget dependency graphs
- Cross-dataset comparison previews before registration

### 13.2 Potential Enhancements

- Column name normalization suggestions
- Data quality scoring
- Schema evolution detection
- Multi-file dataset inference

---

## 14. Summary

Datasculpt is **structured skepticism applied to data**.

It does not guess meaning. It clarifies structure. And it gives Invariant the footing it needs to reason correctly.

The key insight: **inference is not about being right, it's about being explicit**. By surfacing assumptions and evidence, Datasculpt enables human oversight while automating the tedious parts of data onboarding.
