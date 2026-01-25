# Documentation Structure for *Datasculpt*

---

## 0. Root Entry Points

### `/README.md` (GitHub-facing, short)

**Purpose:** Fast orientation + funnel into docs

* What Datasculpt is (1 sentence)
* What problem it solves (3 bullets)
* What it is *not*
* 3 links only:
  * Quickstart
  * Examples
  * Concepts

---

### `/docs/index.md` — **Documentation Home**

**Purpose:** Reader map + navigation

* What this documentation is for
* Who should read which sections
* Documentation map (diagram or table)
* Links to:
  * Getting Started
  * Examples
  * Concepts
  * Integration
  * Reference

---

## 1. Getting Started (success-first)

### `/docs/getting-started/index.md`

**Purpose:** Onboarding path overview

* What you'll learn here
* Prerequisites (minimal: Python 3.11+, pandas)
* Recommended reading order

---

### `/docs/getting-started/quickstart.md`

**Purpose:** First successful inference in 5 minutes

Sections:

1. What you're about to do
2. Install datasculpt (`pip install datasculpt`)
3. Run inference on a sample CSV
4. Examine the output (shape, grain, roles)
5. See the decision record (why each choice was made)
6. Try an ambiguous dataset (see questions generated)
7. What just happened (very high level)

---

### `/docs/getting-started/mental-model.md`

**Purpose:** Minimal conceptual framing

* Where Datasculpt sits in your stack (upstream of catalogs, semantic layers)
* "Evidence → Hypotheses → Decision" pipeline
* The three outputs: shape, grain, column roles
* Why determinism matters

---

## 2. Examples (learn by inference)

### `/docs/examples/index.md`

**Purpose:** Scenario-driven learning

* What these examples show
* How to read them (input → inference → result)
* Link list

---

### `/docs/examples/wide-observations.md`

* Scenario: Spreadsheet-style data with measures as columns
* What the data looks like
* How Datasculpt identifies dimensions vs measures
* Grain inference from natural keys
* Typical output

---

### `/docs/examples/long-indicators.md`

* Scenario: Unpivoted data with indicator_name/value columns
* Detection of indicator pattern
* Why this shape matters for downstream aggregation safety
* Typical output

---

### `/docs/examples/wide-time-columns.md`

* Scenario: Time periods encoded in column headers (2020, 2021, 2022...)
* Header date detection
* Structural implications for time series analysis
* Typical output

---

### `/docs/examples/series-column.md`

* Scenario: JSON arrays or objects storing time series in a single column
* Array/object detection in values
* When to use this pattern
* Typical output

---

### `/docs/examples/ambiguous-shape.md`

* Scenario: Dataset that could reasonably be long or wide
* How Datasculpt handles low confidence gaps
* Interactive mode: questions generated
* Applying user answers to resolve ambiguity

---

### `/docs/examples/grain-detection.md`

* Scenario: Finding the minimal unique key
* Single-column vs composite keys
* Pseudo-key detection (row_id, uuid) and penalties
* Survey-aware patterns (person_id, member_id)

---

## 3. Concepts (domain understanding)

### `/docs/concepts/index.md`

**Purpose:** Conceptual foundation

* Why structural metadata is a missing category
* How Datasculpt produces understanding (not assumes it)
* When to read this section

---

### `/docs/concepts/structural-metadata.md`

* The five metadata categories (technical, business, operational, governance, **structural**)
* What structural metadata captures that others don't
* Why it must be inferred before registration

---

### `/docs/concepts/evidence.md`

* What ColumnEvidence captures
* Primitive types vs structural types
* Statistics: null rate, cardinality, value profiles
* Parse results: dates, JSON, arrays
* Why evidence is separate from interpretation

---

### `/docs/concepts/shapes.md`

* The five shape hypotheses
* `long_observations` — rows as atomic observations
* `long_indicators` — unpivoted indicator/value pairs
* `wide_observations` — measures as columns
* `wide_time_columns` — time in headers
* `series_column` — arrays/objects in cells
* How shapes are scored and ranked

---

### `/docs/concepts/roles.md`

* The eight column roles
* `key` — uniqueness contributor
* `dimension` — categorical grouping
* `measure` — numeric aggregatable
* `time` — temporal dimension
* `indicator_name` / `value` — unpivoted indicator pairs
* `series` — embedded time series
* `metadata` — descriptive, non-analytical
* Multi-candidate scoring with confidence

---

### `/docs/concepts/grain.md`

* What grain is (minimal unique key set)
* Why most data errors are grain errors
* Candidate testing: singles first, then combinations
* Pseudo-key penalties
* Uniqueness ratio and confidence
* Grain as first-class artifact

---

### `/docs/concepts/decision-records.md`

* Anatomy of a DecisionRecord
* Selected hypothesis + ranked alternatives
* Evidence trail for each decision
* Pending questions for ambiguous aspects
* Why auditability matters

---

## 4. Architecture & Design (how it works)

### `/docs/architecture/index.md`

**Purpose:** System-level understanding

* Design goals (determinism, evidence-based, focused)
* Non-goals (not a catalog, not ETL, not semantic identity)
* Where complexity lives

---

### `/docs/architecture/pipeline.md`

* The inference pipeline:
  ```
  Input → Evidence → Roles → Shape → Grain → Questions → Decision → Proposal
  ```
* Module responsibilities
* Data flow between stages

---

### `/docs/architecture/design-principles.md`

* **Determinism first** — identical input → identical output
* **Evidence, not authority** — every inference scored and justified
* **Shape before semantics** — structure first, meaning later
* **Minimal core** — only pandas required
* **Multi-candidate scoring** — rank alternatives, surface ambiguity

---

### `/docs/architecture/scope-boundaries.md`

* What is in scope (shape, roles, grain, evidence)
* What is explicitly out of scope (semantic identity, data quality, transformation)
* Anti-scope-creep checklist
* Where Datasculpt ends and Invariant begins

---

## 5. Integration Guide (how to use it)

### `/docs/integration/index.md`

**Purpose:** Adoption path

* What you need to integrate
* What you can ignore initially
* Recommended order

---

### `/docs/integration/minimal-integration.md`

* Core inference only
* `infer(filepath)` → `InferenceResult`
* Extracting shape, grain, roles from result
* No optional dependencies

---

### `/docs/integration/interactive-mode.md`

* When ambiguity requires human input
* Generating questions (`interactive=True`)
* Collecting answers
* Applying answers (`apply_answers()`)
* Cumulative answer merging

---

### `/docs/integration/optional-adapters.md`

* Frictionless adapter (schema validation hints)
* DataProfiler adapter (statistical profiling)
* YData adapter (HTML reports)
* How to enable: `pip install datasculpt[frictionless]`

---

### `/docs/integration/invariant-handoff.md`

* InvariantProposal structure
* Required fields for registration
* Warnings and confirmations
* Linked decision records for auditability

---

## 6. Reference Documentation (lookup, not learning)

### `/docs/reference/index.md`

**Purpose:** Explicitly "for lookup"

* How to use this section
* Stability guarantees

---

### `/docs/reference/glossary.md`

* Full terminology (generated from types)

---

### `/docs/reference/api.md`

* `infer(filepath, config, interactive)` → `InferenceResult`
* `apply_answers(result, answers)` → `InferenceResult`
* `extract_dataframe_evidence(df)` → `Dict[str, ColumnEvidence]`
* `detect_shape(evidence, config)` → `ShapeResult`
* `infer_grain(df, evidence, config)` → `GrainInference`

---

### `/docs/reference/types.md`

* `ColumnEvidence`
* `ShapeHypothesis` (enum)
* `ShapeResult`
* `Role` (enum)
* `RoleScore`
* `GrainInference`
* `DecisionRecord`
* `InvariantProposal`
* `InferenceResult`

---

### `/docs/reference/configuration.md`

* `InferenceConfig` fields
* Role scoring thresholds
* Shape detection sensitivity
* Grain inference limits
* Ambiguity detection parameters

---

### `/docs/reference/cli.md`

* `datasculpt infer <filepath>` — run inference
* `datasculpt preview <filepath>` — preview without full inference
* Output formats (JSON, YAML, human-readable)
* Exit codes

---

## 7. Advanced Usage

### `/docs/advanced/index.md`

**Purpose:** Power user scenarios

---

### `/docs/advanced/custom-configuration.md`

* Tuning thresholds for specific domains
* Conservative vs aggressive settings
* Domain-specific role patterns

---

### `/docs/advanced/step-by-step.md`

* Running pipeline stages independently
* Evidence extraction only
* Role scoring with custom evidence
* Shape detection without grain

---

### `/docs/advanced/extending-roles.md`

* Adding domain-specific role patterns
* Custom regex for column name matching
* Value-based role detection

---

### `/docs/advanced/special-columns.md`

* Weight columns (survey weights, sample weights)
* Denominator columns (rate calculations)
* Suppression flags (disclosure control)
* Quality flags (data quality indicators)
* Multi-candidate detection and confirmation

---

## 8. Troubleshooting & FAQ

### `/docs/troubleshooting/index.md`

**Purpose:** Problem-solving guide

---

### `/docs/troubleshooting/low-confidence.md`

* What low confidence means
* Reading the decision record
* Providing manual answers
* When to trust vs override

---

### `/docs/troubleshooting/wrong-shape.md`

* Common misdetections
* How to diagnose from evidence
* Interactive mode for correction
* Providing shape hints

---

### `/docs/troubleshooting/grain-issues.md`

* Duplicate keys detected
* No unique grain found
* Pseudo-key penalties explained
* Manual grain specification

---

### `/docs/troubleshooting/performance.md`

* Large file handling
* Sampling configuration
* Memory considerations
* Optional adapter overhead

---

## 9. Development & Contributing

### `/docs/development/index.md`

**Purpose:** Contributor guide

---

### `/docs/development/testing.md`

* Test organization (unit, integration, fixtures)
* Key test properties (determinism, order invariance)
* Running tests
* Adding new fixtures

---

### `/docs/development/adding-shapes.md`

* Adding a new shape hypothesis
* Scoring function requirements
* Integration with existing pipeline

---

### `/docs/development/adding-roles.md`

* Adding a new column role
* Pattern matching additions
* Type-based signals
* Testing role detection

---

### `/docs/development/adapters.md`

* Adapter architecture
* Creating a new adapter
* Optional dependency management
* Testing adapters

---

## 10. Appendices

### `/docs/appendix/design-constraints.md`

* The "constitution" — non-negotiable rules
* Why determinism is mandatory
* Why LLMs are advisory only
* Why evidence trails are required

---

### `/docs/appendix/shape-decision-tree.md`

* Visual decision tree for shape detection
* Scoring algorithm walkthrough
* Edge cases and tiebreakers

---

### `/docs/appendix/role-heuristics.md`

* Complete heuristic rules for each role
* Pattern lists
* Cardinality thresholds
* Type requirements

---

### `/docs/appendix/fixtures.md`

* Canonical test fixtures
* What each fixture demonstrates
* Expected inference results

---

## Final note (structural intent)

This outline enforces:

* **Orientation before rigor** — quickstart shows value immediately
* **Examples before philosophy** — see inference work before understanding why
* **Integration before architecture** — use it before understanding internals
* **Reference clearly separated from learning** — lookup tables vs tutorials

**Highest leverage next steps:**

1. Write `/docs/index.md` — the reader map
2. Write `/getting-started/quickstart.md` — first 5-minute success
3. Add 3 killer examples (wide observations, long indicators, ambiguous shape)

Those three alone will radically change adoption.
