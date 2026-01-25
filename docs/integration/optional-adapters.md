# Optional Adapters

Enhanced profiling capabilities through optional dependencies.

## Available Adapters

| Adapter | Package | Purpose |
|---------|---------|---------|
| Frictionless | `frictionless` | Schema validation hints |
| DataProfiler | `dataprofiler` | Statistical profiling |
| YData | `ydata-profiling` | HTML profile reports |

## Installation

```bash
# Individual adapters
pip install datasculpt[frictionless]
pip install datasculpt[dataprofiler]
pip install datasculpt[reporting]

# All adapters
pip install datasculpt[all]
```

## Enabling Adapters

```python
from datasculpt.core.types import InferenceConfig

config = InferenceConfig(
    use_frictionless=True,
    use_dataprofiler=True,
)

result = infer("data.csv", config=config)
```

## Frictionless Adapter

[Frictionless](https://framework.frictionlessdata.io/) provides schema inference and validation.

### What It Adds

- Schema type inference
- Format detection (CSV dialect, encoding)
- Constraint detection (unique, required, enum)
- Field descriptions

### Usage

```python
config = InferenceConfig(use_frictionless=True)
result = infer("data.csv", config=config)

# Frictionless data in external field
ev = result.decision_record.column_evidence["geo_id"]
>>> ev.external.get("frictionless")
{
    'type': 'string',
    'format': 'default',
    'constraints': {'required': True},
}
```

### Installation

```bash
pip install datasculpt[frictionless]
# or
pip install frictionless>=5.0
```

## DataProfiler Adapter

[DataProfiler](https://github.com/capitalone/dataprofiler) provides deep statistical analysis.

### What It Adds

- Detailed statistics (percentiles, histograms)
- Data type detection with confidence
- Sensitive data detection (PII, credentials)
- Data quality metrics

### Usage

```python
config = InferenceConfig(use_dataprofiler=True)
result = infer("data.csv", config=config)

# DataProfiler data in external field
ev = result.decision_record.column_evidence["population"]
>>> ev.external.get("dataprofiler")
{
    'data_type': 'int',
    'data_type_ratio': 1.0,
    'statistics': {
        'min': 580000,
        'max': 1300000,
        'mean': 875000,
        'stddev': 285000,
        'histogram': {...}
    }
}
```

### Installation

```bash
pip install datasculpt[dataprofiler]
# or
pip install dataprofiler>=0.12
```

## YData Profiling Adapter

[YData Profiling](https://github.com/ydataai/ydata-profiling) generates comprehensive HTML reports.

### What It Adds

- Interactive HTML reports
- Correlation analysis
- Missing value analysis
- Sample data display
- Alerts for data quality issues

### Usage

```python
from datasculpt.adapters.reporting import generate_profile_report

# Generate HTML report
generate_profile_report(
    df=result.dataframe,
    output_path="profile_report.html",
    title="Data Profile Report"
)
```

### Installation

```bash
pip install datasculpt[reporting]
# or
pip install ydata-profiling>=4.0
```

## Adapter Architecture

Adapters are optional and isolated:

```
datasculpt/
├── core/           # No optional dependencies
│   ├── evidence.py
│   ├── roles.py
│   └── ...
└── adapters/       # Optional dependencies
    ├── frictionless_adapter.py
    ├── dataprofiler_adapter.py
    └── reporting.py
```

If an adapter is enabled but not installed, you get a clear error:

```python
config = InferenceConfig(use_frictionless=True)
result = infer("data.csv", config=config)
# ImportError: Frictionless adapter requires: pip install datasculpt[frictionless]
```

## Combining Adapters

Use multiple adapters together:

```python
config = InferenceConfig(
    use_frictionless=True,
    use_dataprofiler=True,
)

result = infer("data.csv", config=config)

# Both sources available
ev = result.decision_record.column_evidence["population"]
>>> ev.external.keys()
dict_keys(['frictionless', 'dataprofiler'])
```

## Performance Considerations

| Adapter | Overhead | When to Use |
|---------|----------|-------------|
| Core only | Minimal | Always, production |
| Frictionless | Low | When schema validation matters |
| DataProfiler | Medium | Deep analysis, data quality |
| YData | High | Reports, exploration |

For production pipelines, consider:
1. Use core only for fast inference
2. Run adapters asynchronously if needed
3. Cache adapter results for repeated access

## Custom Adapters

Create your own adapter:

```python
# myproject/adapters/custom.py
def extract_custom_evidence(df, column_name):
    """Extract custom evidence for a column."""
    series = df[column_name]
    return {
        "custom_metric": calculate_custom_metric(series),
        "custom_flag": detect_custom_pattern(series),
    }

# Usage in your pipeline
from datasculpt.core.evidence import extract_dataframe_evidence
from myproject.adapters.custom import extract_custom_evidence

evidence = extract_dataframe_evidence(df)
for col_name in df.columns:
    evidence[col_name].external["custom"] = extract_custom_evidence(df, col_name)
```

## Next Steps

- [Invariant Handoff](invariant-handoff.md) — Downstream integration
- [API Reference](../reference/api.md) — Full function signatures
