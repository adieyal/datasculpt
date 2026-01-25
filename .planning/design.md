Below are **both artifacts**, ready to drop into a repo and to hand to an implementing LLM.

---

# Part A — Datasculpt Design Document (Markdown)

You can save this verbatim as:

```
docs/design/datasculpt.md
```

---

## Datasculpt

**Deterministic Dataset Shape & Semantic Inference for Invariant**

### Status

Draft — implementation-ready

### Owner

Invariant / Data Infrastructure

---

## 1. Overview

Datasculpt is a standalone library responsible for **inferring the structural intent of tabular datasets** prior to registration in Invariant.

It answers questions such as:

* What *shape* is this dataset (long, wide, time-in-headers, series-object)?
* What *roles* do columns play (key, dimension, measure, time, series)?
* What is the likely *grain* of observation?
* What assumptions are being made, and how confident are they?

Datasculpt **does not decide meaning**.
It produces **evidence-backed proposals** that Invariant can validate, enrich, or reject.

---

## 2. Problem Context

Most datasets arrive as CSVs or spreadsheets with:

* arbitrary column names
* inconsistent formatting
* no schema or metadata

Traditional ETL tools focus on *movement* of data, not *understanding* it.

Invariant requires structural clarity to reason about:

* dataset compatibility
* aggregation correctness
* semantic equivalence

Datasculpt fills the gap between “raw table” and “semantic system”.

---

## 3. Design Principles

### 3.1 Determinism

Given the same input and configuration, Datasculpt must produce the same output.

No hidden randomness.
No LLMs in the decision loop.

---

### 3.2 Evidence, Not Authority

Datasculpt never asserts truth.

Every inference is:

* scored
* justified
* reversible

Ambiguity is surfaced explicitly.

---

### 3.3 Shape Before Semantics

Datasculpt focuses on *structure*:

* roles
* grain
* layout

Semantic identity (what a variable *means*) belongs to Invariant’s Identity and Semantic layers.

---

### 3.4 Minimal Core Dependencies

Core inference relies on:

* pandas
* Python standard library

Heavier profilers are **optional plugins**, never required.

---

## 4. Scope & Non-Goals

### In Scope

* Column-level evidence extraction
* Dataset shape inference
* Grain (key) inference
* Interactive clarification
* Invariant registration proposal

### Out of Scope

* ETL orchestration
* Ontology resolution
* Automated semantic equivalence
* Data storage or querying

---

## 5. High-Level Architecture

```
Tabular Input
   │
   ▼
Evidence Extraction (deterministic)
   │
   ▼
Role Scoring (per column)
   │
   ▼
Shape Hypothesis Scoring (dataset-level)
   │
   ▼
Grain Inference
   │
   ▼
Decision Record + Questions
   │
   ▼
Invariant Proposal
```

Optional plugins can **add evidence**, but never override decisions.

---

## 6. Core Concepts

### 6.1 Column Evidence

A normalized, internal representation of everything known about a column:

* primitive type
* structural type (scalar / array)
* parse success rates
* statistical properties
* role likelihoods

This isolates the system from profiler-specific schemas.

---

### 6.2 Shape Hypotheses

Supported hypotheses:

* `long_observations`
* `long_indicators`
* `wide_observations`
* `wide_time_columns`
* `series_column`

Each hypothesis is scored independently using column evidence.

---

### 6.3 Grain

The grain is the **minimal set of columns that uniquely identify an observation**.

Datasculpt infers candidate grains by:

* evaluating combinations of high-cardinality columns
* measuring uniqueness ratios
* preferring smaller, more stable keys

If no strong grain exists, this is surfaced as a warning.

---

### 6.4 Decision Record

Every run produces a decision record capturing:

* evidence
* rejected hypotheses
* confidence levels
* user interventions (if any)

This record is persisted for auditability.

---

## 7. Interactive Inference

When ambiguity remains:

* Datasculpt generates structured questions
* Users (or an LLM acting as an assistant) respond
* The pipeline is re-run with constraints applied

LLMs are advisory only; deterministic logic remains authoritative.

---

## 8. Mapping to Invariant

Datasculpt outputs an **InvariantProposal**, not direct writes.

Invariant is responsible for:

* catalog persistence
* semantic validation
* reference system resolution
* compatibility checks

Datasculpt provides the structural scaffolding.

---

## 9. Configuration

All thresholds are configurable (e.g. parse rates, uniqueness thresholds) but defaults are conservative and explicit.

---

## 10. Testing Strategy

* Unit tests for evidence extraction
* Integration tests on canonical fixtures
* Contract tests for InvariantProposal validity
* Property tests (order invariance, naming robustness)

---

## 11. Success Criteria

Datasculpt is successful if:

* most datasets register without hand-written schemas
* assumptions are visible and reviewable
* Invariant compatibility errors decrease
* users trust the system because it explains itself

---

## 12. Summary

Datasculpt is **structured skepticism applied to data**.

It does not guess meaning.
It clarifies structure.
And it gives Invariant the footing it needs to reason correctly.

---

# Part B — JSON Schemas (Machine-Readable Contracts)

These schemas are designed to:

* validate LLM output
* enforce determinism
* protect Invariant from malformed proposals

You can store them as:

```
schemas/invariant_proposal.schema.json
schemas/decision_record.schema.json
```

---

## B1. `InvariantProposal` JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://invariant.dev/schemas/invariant_proposal.json",
  "title": "InvariantProposal",
  "type": "object",
  "required": [
    "dataset_name",
    "dataset_kind",
    "grain",
    "columns",
    "shape_hypothesis",
    "decision_record_id"
  ],
  "properties": {
    "dataset_name": {
      "type": "string"
    },
    "dataset_kind": {
      "type": "string",
      "enum": [
        "observations",
        "indicators_long",
        "timeseries_wide",
        "timeseries_series"
      ]
    },
    "shape_hypothesis": {
      "type": "string",
      "enum": [
        "long_observations",
        "long_indicators",
        "wide_observations",
        "wide_time_columns",
        "series_column"
      ]
    },
    "grain": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "columns": {
      "type": "array",
      "minItems": 1,
      "items": {
        "$ref": "#/$defs/ColumnSpec"
      }
    },
    "warnings": {
      "type": "array",
      "items": { "type": "string" }
    },
    "required_user_confirmations": {
      "type": "array",
      "items": { "type": "string" }
    },
    "decision_record_id": {
      "type": "string"
    }
  },
  "$defs": {
    "ColumnSpec": {
      "type": "object",
      "required": [
        "name",
        "role",
        "primitive_type",
        "structural_type"
      ],
      "properties": {
        "name": { "type": "string" },
        "role": {
          "type": "string",
          "enum": [
            "key",
            "dimension",
            "measure",
            "time",
            "indicator_name",
            "value",
            "series",
            "metadata"
          ]
        },
        "primitive_type": {
          "type": "string",
          "enum": [
            "string",
            "integer",
            "number",
            "boolean",
            "date",
            "datetime",
            "unknown"
          ]
        },
        "structural_type": {
          "type": "string",
          "enum": [
            "scalar",
            "array",
            "object",
            "unknown"
          ]
        },
        "reference_system_hint": {
          "type": ["string", "null"]
        },
        "concept_hint": {
          "type": ["string", "null"]
        },
        "unit_hint": {
          "type": ["string", "null"]
        },
        "time_granularity": {
          "type": ["string", "null"]
        },
        "notes": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    }
  }
}
```

---

## B2. `DecisionRecord` JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://invariant.dev/schemas/decision_record.json",
  "title": "DecisionRecord",
  "type": "object",
  "required": [
    "decision_id",
    "dataset_fingerprint",
    "timestamp",
    "selected_hypothesis",
    "hypotheses",
    "grain",
    "column_evidence"
  ],
  "properties": {
    "decision_id": { "type": "string" },
    "dataset_fingerprint": { "type": "string" },
    "timestamp": { "type": "string", "format": "date-time" },
    "selected_hypothesis": {
      "type": "string"
    },
    "hypotheses": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/HypothesisScore"
      }
    },
    "grain": {
      "$ref": "#/$defs/GrainInference"
    },
    "column_evidence": {
      "type": "object",
      "additionalProperties": {
        "$ref": "#/$defs/ColumnEvidence"
      }
    },
    "questions": {
      "type": "array",
      "items": { "$ref": "#/$defs/Question" }
    },
    "answers": {
      "type": "object",
      "additionalProperties": {}
    }
  },
  "$defs": {
    "HypothesisScore": {
      "type": "object",
      "required": ["hypothesis", "score", "reasons"],
      "properties": {
        "hypothesis": { "type": "string" },
        "score": { "type": "number" },
        "reasons": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "GrainInference": {
      "type": "object",
      "required": ["key_columns", "confidence", "uniqueness_ratio"],
      "properties": {
        "key_columns": {
          "type": "array",
          "items": { "type": "string" }
        },
        "confidence": { "type": "number" },
        "uniqueness_ratio": { "type": "number" },
        "evidence": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "ColumnEvidence": {
      "type": "object",
      "required": ["primitive_type", "structural_type", "stats", "parse"],
      "properties": {
        "primitive_type": { "type": "string" },
        "structural_type": { "type": "string" },
        "stats": { "type": "object" },
        "parse": { "type": "object" },
        "role_candidates": {
          "type": "object",
          "additionalProperties": { "type": "number" }
        },
        "notes": {
          "type": "array",
          "items": { "type": "string" }
        },
        "external": {
          "type": "object",
          "additionalProperties": { "type": "object" }
        }
      }
    },
    "Question": {
      "type": "object",
      "required": ["id", "type", "prompt"],
      "properties": {
        "id": { "type": "string" },
        "type": {
          "type": "string",
          "enum": ["choose_one", "choose_many", "confirm", "free_text"]
        },
        "prompt": { "type": "string" },
        "choices": {
          "type": "array",
          "items": { "type": "object" }
        },
        "default": {},
        "rationale": { "type": ["string", "null"] }
      }
    }
  }
}
```

---

## What you now have

* A **human-readable design doc** that explains *why* and *how* Datasculpt works
* **Strict JSON schemas** to:

  * validate LLM output
  * lock down contracts
  * protect Invariant from drift

This is enough to:

* hand to an implementing LLM
* review PRs against
* build Datasculpt and integrate it cleanly into Invariant

If you want next:

* a **step-by-step implementation plan for an LLM agent**
* or a **checklist-style “LLM instructions” distilled from this**
* or a **sample end-to-end JSON output for one of your fixtures**

Just say where you want to push next.

