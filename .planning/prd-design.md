Below is a **full PRD** for the Datasculpt project, written to be directly usable as a design- and planning-grade artifact. It is intentionally explicit about **decision boundaries**, **non-goals**, and **why certain dependencies are optional**, because that’s the architectural risk you’re managing.

---

# PRD: Datasculpt — Dataset Shape & Semantic Inference for Invariant

## 1. Background & Motivation

Invariant’s core promise is **semantic compatibility**: the ability to reason about whether two datasets, metrics, or columns can be meaningfully compared, joined, or transformed.

This requires more than primitive typing. It requires understanding:

* dataset *shape* (wide vs long vs series objects)
* column *roles* (identifier, dimension, measure, time, series)
* dataset *grain* (the implied grouping of observations)
* basic *semantic intent* (e.g. “this is a time series indexed by month per geography”)

However, most real-world data arrives as **opaque tabular files** (CSV, spreadsheets) with:

* arbitrary column names
* inconsistent formats
* no explicit schema or metadata

Existing ETL tools focus on *moving* data, not *understanding* it.

**Datasculpt exists to fill this gap**: a deterministic, explainable inference layer that transforms raw tabular data into a **structured proposal** that Invariant can accept, validate, or reject.

---

## 2. Problem Statement

Given a tabular dataset of unknown structure:

* We cannot assume column meaning from names alone.
* We cannot assume shape (wide/long/series).
* We cannot assume compatibility with existing datasets.

At the same time:

* Fully automatic semantic inference is neither reliable nor desirable.
* Humans should be involved — but guided by strong defaults and evidence.

The problem is to **infer the shape and intent of a dataset well enough to register it with Invariant**, while keeping the process:

* deterministic
* auditable
* minimally dependent on heavyweight libraries
* extensible over time

---

## 3. Goals & Non-Goals

### Goals

* Infer dataset **shape**, **column roles**, and **candidate grain**
* Produce a **registration proposal** suitable for Invariant
* Support **interactive inference**, including LLM assistance
* Be usable as a **standalone library**
* Keep core dependencies minimal and stable

### Non-Goals

* Full ETL orchestration or scheduling
* Automated semantic ontology resolution (e.g. “this is GDP”)
* Perfect inference without user confirmation
* Acting as a data warehouse or transformation engine

---

## 4. Design Principles & Decisions

### 4.1 Core vs Optional Dependencies (Decision)

**Decision:** Datasculpt core will not depend on heavy profiling libraries.

**Rationale:**

* Frictionless and DataProfiler provide useful signals, but:

  * they do not infer roles, shape, or grain
  * they introduce instability and heavy transitive dependencies
* Core signals can be computed deterministically with pandas + stdlib

**Resulting architecture:**

* `datasculpt-core`: minimal, deterministic
* optional adapters:

  * `datasculpt-frictionless`
  * `datasculpt-dataprofiler`
  * `datasculpt-reporting` (HTML profiling)

---

### 4.2 Evidence, Not Truth (Decision)

**Decision:** Datasculpt does not “decide” semantics; it produces **evidence-backed hypotheses**.

Every inference is:

* explainable (“why this shape?”)
* reversible
* open to user confirmation or override

This avoids hidden magic and aligns with Invariant’s auditability goals.

---

### 4.3 Shape-First, Semantics-Later (Decision)

**Decision:** Datasculpt focuses on *structural understanding*, not deep domain semantics.

Invariant (and downstream systems) handle:

* ontology alignment
* cross-dataset compatibility
* reference systems

Datasculpt’s job ends once the dataset’s *structural intent* is clear.

---

## 5. Core Concepts

### 5.1 ColumnEvidence

A normalized set of facts about a column, derived from:

* primitive type inference
* parse success rates (date, JSON array, numeric)
* statistics (null rate, distinct ratio)
* header parsing (date-like headers)

This abstracts away differences between profiling libraries.

---

### 5.2 Shape Hypotheses

Datasculpt evaluates competing hypotheses, e.g.:

* **Long Observations**

  * rows are atomic observations
  * dimensions + measures
* **Wide Observations**

  * measures as columns
* **Wide Time Columns**

  * time encoded in headers
* **Series Column**

  * time series stored as arrays or objects

Each hypothesis is scored based on ColumnEvidence.

---

### 5.3 Decision Record

Every inference produces a structured decision record:

* accepted hypothesis
* rejected hypotheses (with reasons)
* confidence score
* evidence summary

This record is persisted and passed to Invariant.

---

## 6. User Stories

### US-1: Basic Dataset Inference

> As a user, I want to provide a CSV and get a structured explanation of its shape and columns so I understand how it will be interpreted.

**Acceptance criteria:**

* system reports dataset shape
* lists column roles with confidence
* shows inferred grain

---

### US-2: Invariant Registration Proposal

> As a system integrator, I want Datasculpt to produce a proposal that Invariant can validate or accept.

**Acceptance criteria:**

* output maps cleanly to Invariant dataset/column models
* no implicit decisions
* all assumptions are explicit

---

### US-3: Interactive Refinement

> As a user, I want to confirm or correct ambiguous inferences.

**Acceptance criteria:**

* ambiguous columns flagged
* user can override roles or shape
* final proposal reflects user choices

---

### US-4: Optional Enhanced Profiling

> As a power user, I want to enable richer profiling without bloating the core system.

**Acceptance criteria:**

* optional installation of profiling adapters
* core behavior unchanged if adapters absent

---

### US-5: LLM-Assisted Interpretation (Optional)

> As a user, I want LLM assistance to explain and resolve ambiguity.

**Acceptance criteria:**

* LLM suggestions are advisory, not authoritative
* deterministic logic remains the source of truth

---

## 7. Functional Requirements

### Core

* Load tabular data (via pandas)
* Compute ColumnEvidence
* Detect dataset shape
* Infer candidate grain
* Produce Invariant-compatible proposal
* Serialize decision record

### Optional

* Integrate Frictionless for schema hints
* Integrate DataProfiler for richer stats
* Generate human-readable profiling reports
* LLM-based explanation and suggestion

---

## 8. Non-Functional Requirements

* Deterministic output for identical input
* Clear error messages and diagnostics
* Minimal dependency footprint
* Fast execution on medium datasets (≤1M rows)
* Fully testable without optional dependencies

---

## 9. Open Questions & Future Work

* Support for nested/object columns beyond arrays
* Incremental inference for streaming data
* Integration with recipe/budget dependency graphs
* Cross-dataset comparison previews before registration

---

## 10. Success Criteria

Datasculpt is successful if:

* Most datasets can be registered with Invariant **without manual schema writing**
* Users understand *why* a dataset was interpreted a certain way
* Invariant compatibility errors decrease due to better upfront structure

---

## 11. Summary

Datasculpt is not “smart ETL.”
It is **structured skepticism applied to tabular data**.

By focusing on shape, evidence, and explicit decisions — and by keeping heavy dependencies optional — it enables Invariant to do what it does best: reason about meaning *after* structure is understood.

If you want, next we can:

* derive a minimal public API from this PRD
* or map this PRD directly onto a module/package structure
* or write the Invariant-side PRD that consumes Datasculpt’s output

