# Changelog

## Stability Guarantees

Datasculpt follows semantic versioning. This table shows the stability of different components:

| Component | Stability | Notes |
|-----------|-----------|-------|
| `infer()` function | **Stable** | Signature and return type will not change in minor versions |
| `InferenceResult` | **Stable** | Core attributes (`shape`, `roles`, `grain`) stable |
| `DecisionRecord` | **Stable** | Structure stable, new fields may be added |
| Shape vocabulary | **Stable** | `wide_observations`, `long_indicators`, `wide_time_columns`, `series_column`, `microdata` |
| Role vocabulary | **Evolving** | New roles may be added; existing roles stable |
| Evidence types | **Evolving** | New evidence types may be added |
| Internal APIs | **Unstable** | Modules under `_internal` may change without notice |
| CLI interface | **Evolving** | Commands may be added or modified |

### What "Stable" Means

- Breaking changes only in major versions (1.0 → 2.0)
- Additions (new optional parameters, new attributes) in minor versions
- Bug fixes in patch versions

### What "Evolving" Means

- May change in minor versions (0.1 → 0.2)
- We aim for backwards compatibility but don't guarantee it
- Changes will be documented in this changelog

## Versioning Policy

Datasculpt uses [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes to stable APIs
- **MINOR** (0.1.0): New features, non-breaking changes
- **PATCH** (0.0.1): Bug fixes, documentation updates

During the 0.x phase, the API is still being refined. We aim for stability but reserve the right to make breaking changes in minor versions when necessary for the library's long-term health.

---

## Releases

### Unreleased

**Features:**

- **Microdata shape detection:** New `microdata` shape hypothesis for survey/observation datasets (DHS, LSMS, MICS style)
  - Detects coded question columns (s1aq1, v101, hv001)
  - Identifies hierarchical ID structure (hhid + indiv)
  - Recognizes geography hierarchies (zone, state, lga)
  - Finds survey weight columns

- **New role values for survey data:**
  - `respondent_id`: Primary unit identifier (hhid, person_id)
  - `subunit_id`: Secondary ID within unit (indiv, member_num)
  - `cluster_id`: Sampling cluster/EA identifier
  - `survey_weight`: Sampling weights
  - `question_response`: Coded survey question answers
  - `geography_level`: Administrative hierarchy levels

- **Evidence contract expansion:**
  - `ColumnEvidence` now includes `n_rows`, `n_non_null`, `top_values`, `value_length_stats`
  - New `ColumnSample` dataclass for deterministic value sampling
  - `InferenceConfig` supports `return_samples`, `sample_size`, `sample_seed` options
  - `InferenceResult` includes optional `column_samples` dict

- **New microdata module:** `datasculpt.core.microdata` with survey-specific detection functions
  - `MicrodataLevel` enum (household, individual, episode, item)
  - `MicrodataProfile` and `QuestionColumnProfile` dataclasses

**Bug Fixes:**

- Fix primitive type detection for pandas 2.x string dtype. Boolean, integer, and date strings are now correctly detected when pandas uses the newer `str` dtype instead of `object`.

---

### v0.1.0 (Current)

*Initial public release*

**Features:**

- Core inference pipeline: evidence extraction → shape detection → role assignment → grain inference
- Four supported shapes: `wide_observations`, `long_indicators`, `wide_time_columns`, `series_column`
- Deterministic inference with no external model calls
- Decision records capturing all evidence and reasoning
- Interactive mode for human-in-the-loop disambiguation
- InvariantProposal output for catalog integration
- CLI for command-line usage
- File format support: CSV, Excel, Parquet

**Known Limitations:**

- No support for nested/hierarchical data
- No streaming support for large files
- Limited to single-table inference

---

## Planned

### v0.2.0

*Planned improvements based on user feedback*

- **Enhanced confidence scoring:** More granular confidence metrics
- **Additional file formats:** JSON lines, SQLite tables
- **Batch processing:** Process multiple files with consistent settings
- **Performance improvements:** Lazy loading for large files

### Future

- Multi-table relationship inference
- Custom role definitions
- Plugin system for evidence extractors
