# Architecture

System-level understanding of Datasculpt.

## Design Goals

1. **Determinism** — Same input → same output
2. **Evidence-based** — Every inference is justified
3. **Focused** — Structure only, not semantics
4. **Minimal dependencies** — Core requires only pandas
5. **Auditable** — Complete decision trails

## Non-Goals

Datasculpt explicitly does **not**:

- **Replace data catalogs** — It produces metadata for catalogs, not stores it
- **Perform ETL** — It analyzes structure, not transforms data
- **Define semantics** — That's Invariant's job
- **Use LLMs for decisions** — Determinism requires rule-based inference
- **Handle data quality** — It reports issues, not fixes them

## Architecture Guides

| Guide | Focus |
|-------|-------|
| [Pipeline](pipeline.md) | Inference stages and data flow |
| [Design Principles](design-principles.md) | Core principles and rationale |
| [Scope Boundaries](scope-boundaries.md) | What's in and out of scope |
