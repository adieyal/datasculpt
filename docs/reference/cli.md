# CLI Reference

Command-line interface for Datasculpt.

## Installation

The CLI is included with the package:

```bash
pip install datasculpt
```

## Commands

### `datasculpt infer`

Run inference on a file.

```bash
datasculpt infer <filepath> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `<filepath>` | Path to CSV file |

**Options:**

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output format: `json`, `yaml`, `text` (default: `text`) |
| `--interactive`, `-i` | Enable interactive mode |
| `--config`, `-c` | Path to config file |
| `--verbose`, `-v` | Show detailed output |

**Examples:**

```bash
# Basic inference
datasculpt infer data.csv

# JSON output
datasculpt infer data.csv -o json

# Interactive mode
datasculpt infer data.csv -i

# With config file
datasculpt infer data.csv -c config.yaml
```

**Output (text):**

```
Shape: wide_observations
Grain: ['geo_id', 'sex', 'age_group']
Uniqueness: 100.0%
Confidence: 0.95

Columns:
  geo_id: dimension (string)
  sex: dimension (string)
  age_group: dimension (string)
  population: measure (integer)
  unemployed: measure (integer)
  unemployment_rate: measure (number)
```

**Output (JSON):**

```json
{
  "shape": "wide_observations",
  "grain": {
    "columns": ["geo_id", "sex", "age_group"],
    "uniqueness_ratio": 1.0,
    "confidence": 0.95
  },
  "columns": [
    {"name": "geo_id", "role": "dimension", "type": "string"},
    {"name": "sex", "role": "dimension", "type": "string"}
  ]
}
```

---

### `datasculpt preview`

Preview a file without full inference.

```bash
datasculpt preview <filepath> [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--rows`, `-n` | Number of rows to show (default: 5) |
| `--output`, `-o` | Output format: `json`, `yaml`, `text` (default: `text`) |

**Examples:**

```bash
# Basic preview
datasculpt preview data.csv

# Show 10 rows
datasculpt preview data.csv -n 10
```

**Output:**

```
File: data.csv
Rows: 8
Columns: 6

Columns:
  geo_id: string (2 unique, 0.0% null)
  sex: string (2 unique, 0.0% null)
  age_group: string (2 unique, 0.0% null)
  population: integer (8 unique, 0.0% null)
  unemployed: integer (8 unique, 0.0% null)
  unemployment_rate: number (8 unique, 0.0% null)

Sample:
  geo_id | sex | age_group | population | unemployed | unemployment_rate
  ZA-GP  | F   | 15-24     | 1200000    | 180000     | 0.15
  ZA-WC  | F   | 15-24     | 600000     | 75000      | 0.125
  ...
```

---

### `datasculpt version`

Show version information.

```bash
datasculpt version
```

**Output:**

```
datasculpt 0.1.0
Python 3.11.4
pandas 2.0.3
```

---

## Configuration File

Create a YAML config file:

```yaml
# config.yaml
key_cardinality_threshold: 0.9
dimension_cardinality_max: 0.1
null_rate_threshold: 0.01
min_time_columns_for_wide: 3
max_grain_columns: 4
min_uniqueness_confidence: 0.95
hypothesis_confidence_gap: 0.1
use_frictionless: false
use_dataprofiler: false
```

Use with:

```bash
datasculpt infer data.csv -c config.yaml
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | File not found |
| 3 | Parse error |
| 4 | Configuration error |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATASCULPT_CONFIG` | Default config file path |
| `DATASCULPT_OUTPUT` | Default output format |
| `DATASCULPT_VERBOSE` | Enable verbose output by default |

```bash
export DATASCULPT_OUTPUT=json
datasculpt infer data.csv  # Uses JSON output
```
