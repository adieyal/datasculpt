# Examples

Learn Datasculpt by seeing inference in action on different dataset shapes.

## How to Read These Examples

Each example shows:
1. **The data** — What the CSV looks like
2. **The inference** — What Datasculpt detects
3. **The evidence** — Why it made that choice
4. **The output** — The proposal and decision record

## Dataset Shapes

| Example | Shape | Key Characteristic |
|---------|-------|--------------------|
| [Wide Observations](wide-observations.md) | `wide_observations` | Measures as columns |
| [Long Indicators](long-indicators.md) | `long_indicators` | indicator/value pairs |
| [Wide Time Columns](wide-time-columns.md) | `wide_time_columns` | Time periods in headers |
| [Series Column](series-column.md) | `series_column` | Arrays in cells |
| [Ambiguous Shape](ambiguous-shape.md) | Varies | Close-scoring hypotheses |

## Other Scenarios

| Example | Focus |
|---------|-------|
| [Grain Detection](grain-detection.md) | Finding the unique key |
