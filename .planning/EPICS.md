# Datasculpt Implementation Epics & Tasks

## Overview

This document organizes the user-facing capabilities into implementation epics. Each epic represents a cohesive set of functionality that delivers value independently.

---

## Epic 1: Dataset Intake & File Handling

**Goal**: Accept tabular files and prepare them for inference.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 1.1 | File upload handler | Accept CSV, Excel (.xlsx/.xls), Parquet files | P0 |
| 1.2 | DataFrame normalization | Convert all formats to pandas DataFrame | P0 |
| 1.3 | Dataset fingerprinting | Generate stable content-based fingerprint (hash of schema + sample) | P0 |
| 1.4 | Dataset versioning | Track fingerprint changes across uploads of "same" dataset | P1 |
| 1.5 | Preview generation | Extract row count, column count, sample rows, basic stats | P0 |

### Acceptance Criteria
- [ ] Upload CSV, Excel, Parquet → get DataFrame
- [ ] Same file → same fingerprint
- [ ] Preview shows first N rows, column names, row count

---

## Epic 2: Evidence Extraction

**Goal**: Extract deterministic facts about each column.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 2.1 | Primitive type detection | Infer string/integer/number/boolean/date/datetime | P0 |
| 2.2 | Null rate calculation | Compute missing value ratio per column | P0 |
| 2.3 | Cardinality analysis | Compute distinct ratio (unique/total) | P0 |
| 2.4 | Date parsing attempts | Try parsing string columns as dates, record success rate | P0 |
| 2.5 | JSON array detection | Detect columns containing JSON arrays (for series) | P0 |
| 2.6 | Header date detection | Check if column names look like dates (e.g., "2024-01") | P0 |
| 2.7 | ColumnEvidence assembly | Combine all signals into ColumnEvidence dataclass | P0 |

### Acceptance Criteria
- [ ] Each column produces a ColumnEvidence object
- [ ] Evidence is deterministic (same data → same evidence)
- [ ] Parse success rates accurately reflect parsing attempts

---

## Epic 3: Role Scoring

**Goal**: Score each column for each possible role.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 3.1 | Key role scorer | High cardinality, low nulls, naming patterns | P0 |
| 3.2 | Dimension role scorer | Low-medium cardinality, string type | P0 |
| 3.3 | Measure role scorer | Numeric type, varying values | P0 |
| 3.4 | Time role scorer | Date parsing success, naming patterns | P0 |
| 3.5 | Indicator name scorer | Low cardinality strings (for long format) | P0 |
| 3.6 | Value role scorer | Numeric, paired with indicator column | P0 |
| 3.7 | Series role scorer | JSON array parsing success | P0 |
| 3.8 | Role assignment resolver | Assign primary role based on scores | P0 |
| 3.9 | Confidence calculation | Compute confidence based on score margins | P0 |

### Acceptance Criteria
- [ ] Every column gets scores for all applicable roles
- [ ] Primary role assigned with confidence level
- [ ] Low-confidence assignments flagged for user review

---

## Epic 4: Shape Hypothesis Detection

**Goal**: Determine dataset layout/structure.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 4.1 | Long observations scorer | Standard tidy data pattern | P0 |
| 4.2 | Long indicators scorer | Indicator/value pair pattern | P0 |
| 4.3 | Wide observations scorer | Multiple measures as columns | P0 |
| 4.4 | Wide time columns scorer | Time periods in column headers | P0 |
| 4.5 | Series column scorer | JSON arrays representing time series | P0 |
| 4.6 | Hypothesis comparison | Rank hypotheses, select best with confidence | P0 |
| 4.7 | Explanation generator | Produce human-readable "why this shape" text | P0 |

### Acceptance Criteria
- [ ] Each test fixture correctly identified
- [ ] Explanations reference specific column evidence
- [ ] Ambiguous cases produce low confidence + questions

---

## Epic 5: Grain Inference

**Goal**: Find the minimal unique key set.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 5.1 | Key candidate ranking | Rank columns by key likelihood | P0 |
| 5.2 | Single-column uniqueness test | Check if any single column is unique | P0 |
| 5.3 | Composite key search | Try column combinations up to max_grain_columns | P0 |
| 5.4 | Uniqueness ratio calculation | Measure how "unique" each candidate is | P0 |
| 5.5 | Grain inference result | Return GrainInference with confidence | P0 |
| 5.6 | No-grain warning | Flag datasets with no stable grain | P1 |

### Acceptance Criteria
- [ ] Single-column keys found efficiently
- [ ] Composite keys found when needed
- [ ] "No grain" cases surfaced with warning

---

## Epic 6: Interactive Clarification

**Goal**: Resolve ambiguity through user input.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 6.1 | Question generation | Generate structured questions for ambiguous cases | P0 |
| 6.2 | Shape confirmation question | "Is this wide or long format?" | P0 |
| 6.3 | Role override question | "What role should column X have?" | P0 |
| 6.4 | Grain confirmation question | "Is [col1, col2] the correct grain?" | P0 |
| 6.5 | Time axis question | Granularity, frequency, start date | P1 |
| 6.6 | Answer application | Re-run pipeline with user constraints | P0 |
| 6.7 | Override tracking | Record all user overrides in decision record | P0 |

### Acceptance Criteria
- [ ] Questions generated for low-confidence inferences
- [ ] Answers can be provided and applied
- [ ] Pipeline re-runs with constraints respected

---

## Epic 7: Time Axis Interpretation

**Goal**: Understand temporal structure.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 7.1 | Time granularity detection | Daily, weekly, monthly, quarterly, annual | P1 |
| 7.2 | Header time parsing | Parse "2024-01", "Q1 2024", etc. from headers | P1 |
| 7.3 | Series frequency inference | Detect frequency from array lengths + metadata | P1 |
| 7.4 | Time axis confirmation | User confirms/overrides detected granularity | P1 |
| 7.5 | Time range extraction | Start/end dates for the dataset | P1 |

### Acceptance Criteria
- [ ] Common time formats recognized
- [ ] Granularity inferred with reasonable accuracy
- [ ] User can override time interpretation

---

## Epic 8: Special Column Detection

**Goal**: Identify columns with special semantics.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 8.1 | Weight column detection | Columns that look like sampling weights | P2 |
| 8.2 | Denominator detection | Columns that serve as denominators for rates | P2 |
| 8.3 | Suppression flag detection | Boolean/string columns indicating data suppression | P2 |
| 8.4 | Quality flag detection | Columns indicating data quality/reliability | P2 |
| 8.5 | User flagging interface | Allow user to mark special columns | P1 |

### Acceptance Criteria
- [ ] Common patterns for special columns recognized
- [ ] User can manually flag any column as special
- [ ] Special columns noted in proposal

---

## Epic 9: Decision Record & Audit

**Goal**: Complete audit trail for all inferences.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 9.1 | DecisionRecord generation | Create record for each inference run | P0 |
| 9.2 | Evidence serialization | Serialize all column evidence | P0 |
| 9.3 | Hypothesis serialization | Record all hypotheses with scores | P0 |
| 9.4 | Override tracking | Record user confirmations/overrides | P0 |
| 9.5 | JSON export | Export decision record as JSON | P0 |
| 9.6 | Decision record retrieval | Load and display past decision records | P1 |

### Acceptance Criteria
- [ ] Every inference produces a decision record
- [ ] Records contain all evidence and reasoning
- [ ] Records can be exported and reviewed

---

## Epic 10: Invariant Proposal Generation

**Goal**: Produce registration-ready proposals.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 10.1 | InvariantProposal generation | Create proposal from inference results | P0 |
| 10.2 | ColumnSpec mapping | Map columns to Invariant column format | P0 |
| 10.3 | DatasetKind assignment | Map shape to Invariant dataset kind | P0 |
| 10.4 | Warning generation | List all assumptions and risks | P0 |
| 10.5 | Confirmation requirements | List required user confirmations | P0 |
| 10.6 | JSON schema validation | Validate proposal against schema | P0 |
| 10.7 | Proposal diff | Compare proposals for same dataset (version diff) | P1 |

### Acceptance Criteria
- [ ] Proposals validate against JSON schema
- [ ] Warnings clearly state assumptions
- [ ] Confirmations block registration until acknowledged

---

## Epic 11: Invariant Registration Integration

**Goal**: Register datasets into Invariant catalog.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 11.1 | Catalog registration | Create Dataset + Variables in Invariant | P0 |
| 11.2 | DataProduct creation | Publish DataProduct from dataset | P1 |
| 11.3 | Semantic view registration | Register semantic dataset view | P1 |
| 11.4 | Registration confirmation | User acknowledges before commit | P0 |
| 11.5 | Rollback support | Undo registration if errors occur | P2 |

### Acceptance Criteria
- [ ] Datasets appear in Invariant catalog after registration
- [ ] Registration requires explicit user confirmation
- [ ] Failed registrations don't leave partial state

---

## Epic 12: Cross-Dataset Compatibility

**Goal**: Assess structural compatibility between datasets.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 12.1 | Grain compatibility check | Do datasets share compatible grain keys? | P1 |
| 12.2 | Time axis compatibility | Compatible time representation? | P1 |
| 12.3 | Join explosion warning | Warn if grain mismatch would explode rows | P1 |
| 12.4 | Mapping suggestions | Suggest reference system mappings | P2 |
| 12.5 | Mapping task creation | Create pending mapping tasks in Invariant | P2 |

### Acceptance Criteria
- [ ] Compatibility issues surfaced before joins
- [ ] Clear warnings for grain mismatches
- [ ] Suggestions are advisory, not automatic

---

## Epic 13: Query Readiness

**Goal**: Enable basic queries on registered datasets.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 13.1 | Basic query execution | Group by dimensions, select measures | P1 |
| 13.2 | Filter by dimensions | WHERE clauses on dimension columns | P1 |
| 13.3 | Query explanation | Show how dataset will be queried | P1 |
| 13.4 | Validation warnings | Pre-query validation for ambiguous structure | P1 |

### Acceptance Criteria
- [ ] Simple queries work on registered datasets
- [ ] Query explain shows interpretation
- [ ] Warnings appear before problematic queries

---

## Epic 14: CLI Interface

**Goal**: Command-line access for all functionality.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 14.1 | `datasculpt infer` command | Run inference on a file | P0 |
| 14.2 | `datasculpt preview` command | Show dataset preview | P0 |
| 14.3 | `datasculpt register` command | Register to Invariant | P1 |
| 14.4 | `datasculpt diff` command | Compare two versions | P2 |
| 14.5 | Non-interactive mode | Fail fast on ambiguity | P1 |
| 14.6 | Interactive mode | Prompt for clarifications | P1 |
| 14.7 | JSON output | Machine-readable output for all commands | P0 |

### Acceptance Criteria
- [ ] All core operations available via CLI
- [ ] Non-interactive mode suitable for CI
- [ ] JSON output for scripting

---

## Epic 15: Optional Profiler Adapters

**Goal**: Enhance inference with optional profilers.

### Tasks

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| 15.1 | Frictionless adapter | Import schema hints from Frictionless | P2 |
| 15.2 | DataProfiler adapter | Import statistics from DataProfiler | P2 |
| 15.3 | ydata-profiling adapter | Generate HTML profiling reports | P2 |
| 15.4 | Adapter registration | Plug adapters into evidence extraction | P2 |
| 15.5 | Graceful degradation | Core works without adapters installed | P0 |

### Acceptance Criteria
- [ ] Core inference works with pandas only
- [ ] Adapters add evidence without changing core logic
- [ ] Missing adapters don't cause errors

---

## Implementation Phases

### Phase 1: Core Inference (Epics 1-5, 9-10)
- File handling, evidence extraction, role/shape/grain inference
- Decision records and proposals
- **Deliverable**: `datasculpt.infer(file) → InvariantProposal`

### Phase 2: Interactive & CLI (Epics 6, 14)
- Question generation and answer handling
- CLI commands for inference and preview
- **Deliverable**: Interactive clarification workflow

### Phase 3: Time & Special Columns (Epics 7-8)
- Time axis interpretation
- Special column detection
- **Deliverable**: Richer structural understanding

### Phase 4: Registration Integration (Epic 11)
- Invariant catalog integration
- Registration workflow
- **Deliverable**: End-to-end dataset onboarding

### Phase 5: Advanced Features (Epics 12-13, 15)
- Cross-dataset compatibility
- Query readiness
- Optional profiler adapters
- **Deliverable**: Full system integration

---

## Priority Legend

- **P0**: Required for MVP
- **P1**: Important for usability
- **P2**: Nice to have / can defer
