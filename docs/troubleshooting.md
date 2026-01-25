# Troubleshooting

This page covers common issues, error messages, and debugging strategies when working with Datasculpt.

## Common Errors

### IntakeError

`IntakeError` is raised when Datasculpt cannot load or parse your input file.

**File not found:**
```
IntakeError: File not found: /path/to/data.csv
```
*Solution:* Check that the file path is correct and the file exists.

**Unsupported format:**
```
IntakeError: Unsupported file format: .json
```
*Solution:* Datasculpt supports CSV, Excel (.xlsx, .xls), and Parquet files. Convert your data to a supported format.

**Encoding issues:**
```
IntakeError: Unable to decode file with encoding 'utf-8'
```
*Solution:* Specify the correct encoding when loading:
```python
from datasculpt import infer

result = infer("data.csv", encoding="latin-1")
```

### ValueError in apply_answers

`ValueError` is raised when `apply_answers()` receives invalid or inconsistent answers.

**Invalid shape answer:**
```
ValueError: Invalid shape 'pivot'. Expected one of: wide_observations, long_indicators, wide_time_columns, series_column
```
*Solution:* Use one of the valid shape values from the question's options.

**Mismatched answers:**
```
ValueError: Answer references column 'sales' which does not exist
```
*Solution:* Ensure column names in your answers match the actual column names in the dataset.

## Common Warnings

### Low Confidence

```
Warning: Shape detection confidence is low (0.52). Consider reviewing the decision record.
```

This warning appears when Datasculpt cannot clearly distinguish between shapes. Common causes:

- Dataset has characteristics of multiple shapes
- Too few rows or columns for reliable detection
- Unusual column naming conventions

*Solution:* Review the decision record and consider using interactive mode to provide explicit answers.

### Ambiguous Shape

```
Warning: Ambiguous shape detected. Both 'wide_observations' and 'long_indicators' score similarly.
```

This means the dataset could reasonably be interpreted as multiple shapes.

*Solution:* Use interactive mode to explicitly specify the intended shape:
```python
from datasculpt import infer_interactive

result = infer_interactive("data.csv")
# Datasculpt will ask you to clarify
```

### No Stable Grain

```
Warning: No stable grain detected. Grain columns may not uniquely identify rows.
```

This warning indicates that the detected grain columns have duplicate combinations.

Common causes:

- Missing a grain column (e.g., time period)
- Data quality issues (actual duplicates)
- Aggregated data without unique identifiers

*Solution:* Inspect the grain evidence in the decision record and consider if additional columns should be included.

### High Null Rate

```
Warning: Column 'notes' has high null rate (0.85). Role assignment may be affected.
```

Columns with many null values may have unreliable role assignments.

*Solution:* This is usually informational. If the column is important, consider cleaning the data before inference.

## FAQ

### Why did Datasculpt detect the wrong shape?

Shape detection uses heuristics based on column patterns, value distributions, and naming conventions. It can be fooled by:

1. **Unconventional naming:** If your indicator column is named `metric_type` instead of `indicator`, the pattern may not match.

2. **Mixed data:** Datasets that genuinely combine characteristics of multiple shapes.

3. **Small samples:** With very few columns or rows, patterns are harder to detect.

*Fix:* Use interactive mode and provide explicit answers:
```python
result = infer_interactive("data.csv")
# Answer: "wide_observations"
```

### Why is no clear grain detected?

Grain detection looks for columns whose combination uniquely identifies rows. Failures occur when:

1. **Time period missing:** Many datasets need a date/period column for uniqueness.

2. **True duplicates:** The data actually contains duplicate records.

3. **Derived measures:** Rows representing calculations over other rows (like totals).

*Fix:* Inspect grain evidence:
```python
result = infer("data.csv")
print(result.decision_record.grain_evidence)
```

### Why wasn't my array column detected?

Datasculpt detects array-valued columns (comma-separated, JSON arrays, etc.) but may miss them if:

1. **Inconsistent delimiters:** Some rows use `,` others use `;`
2. **Single-element arrays:** Arrays with only one value look like scalars
3. **Escaped content:** JSON arrays with escaped quotes

*Fix:* Pre-process array columns to a consistent format before inference.

### How do I override a detection?

Use `apply_answers()` to override any detection:

```python
from datasculpt import infer

result = infer("data.csv")

# Override the detected shape
corrected = result.apply_answers({
    "shape": "long_indicators"
})
```

## Debugging Tips

### Inspect the Decision Record

The decision record contains all evidence and reasoning:

```python
result = infer("data.csv")

# View shape decision
print(result.decision_record.shape)
print(f"Confidence: {result.decision_record.shape.confidence}")
print(f"Evidence: {result.decision_record.shape.evidence}")

# View role assignments
for col, role in result.decision_record.roles.items():
    print(f"{col}: {role.assigned} (confidence: {role.confidence})")
```

### Check Column Evidence

Each column has detailed evidence:

```python
result = infer("data.csv")

for col in result.columns:
    evidence = result.evidence[col]
    print(f"\n{col}:")
    print(f"  dtype: {evidence.dtype}")
    print(f"  null_rate: {evidence.null_rate}")
    print(f"  unique_rate: {evidence.unique_rate}")
    print(f"  patterns: {evidence.patterns}")
```

### Enable Verbose Logging

For detailed debugging, enable logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("datasculpt")
logger.setLevel(logging.DEBUG)

result = infer("data.csv")
```

### Export Decision Record

Save the decision record for review or sharing:

```python
result = infer("data.csv")

# As JSON
with open("decision.json", "w") as f:
    f.write(result.decision_record.to_json(indent=2))

# As dict
decision_dict = result.decision_record.to_dict()
```

## Getting Help

If you're stuck:

1. **Check the examples:** [Examples](examples/index.md) cover common patterns
2. **Review concepts:** [Concepts](concepts/index.md) explain the underlying model
3. **File an issue:** [GitHub Issues](https://github.com/adieyal/datasculpt/issues)
