Below is a **positioning and explanation document** you can include alongside the PRD (or even ahead of it) to make Datasculpt legible to engineers, product owners, and future contributors. It deliberately clarifies **what Datasculpt is**, **what problem it uniquely solves**, and **what it explicitly refuses to do**.

You can treat this as a “why this exists” document.

---

# Datasculpt: Structural Metadata Inference as a First-Class Concern

## Executive Summary

Datasculpt exists to solve a problem that nearly every data platform *assumes away*:

> **Before data can be governed, queried, compared, or reasoned about, its structure must be understood.**

Most modern data systems—catalogs, semantic layers, governance engines, schema-evolution tools, and GenAI assistants—*depend* on this understanding. Almost none of them *produce it*.

Datasculpt fills that gap.

It is not a data catalog.
It is not an ontology engine.
It is not an ETL tool.

Datasculpt is a **structural metadata producer**: a system whose sole responsibility is to infer, explain, and record the *structural intent* of tabular datasets before they enter a semantic ecosystem such as Invariant.

---

## The Hidden Assumption in Modern Data Systems

Across the literature on data catalogs, governance, semantic discovery, schema evolution, and AI-assisted data management, a consistent pattern emerges:

> Every downstream system assumes that someone already knows:
>
> * what a row represents
> * which columns identify an observation
> * which columns are measures vs dimensions
> * how time is represented
> * what the grain of the dataset is

These assumptions are rarely written down.
They are embedded in:

* query logic
* metric definitions
* dashboards
* joins
* suppression rules
* governance policies

When these assumptions are wrong—or merely implicit—systems do not fail loudly. They fail *quietly*, by producing misleading results.

Datasculpt exists to surface these assumptions **early, explicitly, and audibly**.

---

## Structural Metadata: A Missing Category

Most data platforms distinguish between:

* **Technical metadata** (schemas, types)
* **Business metadata** (definitions, owners)
* **Operational metadata** (lineage, freshness)
* **Governance metadata** (policies, quality rules)

What Datasculpt produces does not fit cleanly into any of these.

It introduces a missing category:

### Structural Metadata

Structural metadata describes *how a dataset is shaped and how it should be interpreted structurally*, independent of business meaning.

It includes:

* dataset shape (long, wide, time-in-headers, series-object)
* column roles (key, dimension, measure, time, series)
* inferred grain (what uniquely identifies a row)
* structural constraints and invariants
* rejected structural interpretations
* confidence levels and ambiguity

This metadata is **orthogonal** to business semantics.
It is the scaffolding on which semantics depend.

---

## Grain Is the Keystone Concept

Across data lake best practices, governance failures, and schema-evolution discussions, one fact becomes unavoidable:

> **Most data errors are grain errors, not type errors.**

Joins break.
Aggregations mislead.
Metrics drift.
Comparisons silently become invalid.

Yet grain is almost never modeled explicitly.

Datasculpt treats grain as:

* a first-class artifact
* something inferred, not assumed
* something versioned and auditable
* something downstream systems must acknowledge

In Datasculpt, a dataset is not “registered” until its grain is either:

* inferred with high confidence, or
* explicitly marked as ambiguous and acknowledged by a user

This single design choice explains a large portion of Datasculpt’s value.

---

## What Datasculpt Is Trying to Solve

Datasculpt addresses a very specific and constrained problem:

> **Given an arbitrary tabular dataset, infer and explain its structural intent well enough that it can safely enter a semantic system.**

Concretely, Datasculpt aims to:

* determine dataset shape
* assign provisional column roles
* infer candidate grain
* surface ambiguity explicitly
* produce a structured proposal for registration
* record *why* these decisions were made

It is a **registration gate**, not a transformation engine.

Nothing enters Invariant silently.

---

## What Datasculpt Is *Not* Trying to Solve

The literature is full of ambitious systems that fail by pulling meaning too far upstream. Datasculpt draws hard boundaries to avoid that fate.

Datasculpt will **not**:

* infer business meaning or ontology identity
* decide that a column “is GDP” or “is unemployment”
* perform instance-level semantic annotation
* create embeddings as a source of truth
* auto-adapt pipelines or schemas
* act as a catalog, governance engine, or discovery UI
* replace human judgment with probabilistic guesses

These are not future roadmap items.
They are **explicit non-goals**.

Datasculpt’s outputs are inputs to those systems—not replacements for them.

---

## Decision Records as Structural Memory

A key insight from schema-evolution and governance work is that systems track *what changed*, but rarely *what changed structurally*.

Datasculpt introduces **decision records** as durable artifacts that capture:

* the selected structural interpretation
* rejected alternatives and why
* inferred grain and its confidence
* column-level evidence
* user confirmations or overrides

These records are:

* versioned
* diffable
* comparable over time

This enables downstream systems to distinguish:

* harmless schema drift
* risky structural changes
* genuinely breaking shifts in interpretation

Without automating governance, Datasculpt makes it *possible*.

---

## Semantic Density Without Semantic Guessing

Datasculpt acknowledges a subtle but important reality:

Some columns cannot stand alone.

A numeric column with:

* no unit
* no context
* no stable interpretation

is structurally incomplete.

Rather than guessing meaning, Datasculpt introduces a lightweight signal:

* **semantic density / context dependence**

This flags columns that:

* require surrounding structure to be meaningful
* are unlikely to be reusable in isolation

It does not name the meaning.
It simply warns downstream systems to be cautious.

---

## The Role of LLMs (Deliberately Limited)

The recent literature on GenAI in data systems reinforces a useful constraint:

LLMs are excellent at:

* explaining ambiguity
* summarizing evidence
* proposing interpretations

They are poor at:

* determinism
* auditability
* owning structural truth

In Datasculpt:

* LLMs may operate *only* on existing evidence and hypotheses
* they may not introduce new shapes, roles, or grains
* they are advisory, never authoritative

Structural truth remains deterministic and inspectable.

---

## Where Datasculpt Fits

Datasculpt sits **upstream** of:

* data catalogs
* semantic layers
* governance engines
* query planners
* AI assistants

It produces the one thing all of them quietly require but rarely generate:

> **Explicit, auditable structural understanding.**

By occupying this narrow but critical niche, Datasculpt avoids scope creep while unlocking correctness everywhere else.

---

## In One Sentence

**Datasculpt is the system that makes implicit structural assumptions explicit—before they become silent failures.**

Everything else builds on that.

