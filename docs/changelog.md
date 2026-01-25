# Changelog

## Stability Guarantees

Datasculpt follows semantic versioning. This table shows the stability of different components:

| Component | Stability | Notes |
|-----------|-----------|-------|
| `infer()` function | **Stable** | Signature and return type will not change in minor versions |
| `InferenceResult` | **Stable** | Core attributes (`shape`, `roles`, `grain`) stable |
| `DecisionRecord` | **Stable** | Structure stable, new fields may be added |
| Shape vocabulary | **Stable** | `wide_observations`, `long_indicators`, `wide_time_columns`, `series_column` |
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
