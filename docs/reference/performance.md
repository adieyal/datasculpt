# Performance Reference

Performance characteristics, limits, and memory considerations.

## Row Limits and Sampling

Datasculpt uses **sampling** for type detection and evidence extraction to maintain consistent performance regardless of dataset size.

### Column Evidence Sampling

| Setting | Value | Purpose |
|---------|-------|---------|
| Sample size | 1,000 rows | Type detection, date parsing, JSON detection |
| Sampling strategy | 500 head + 500 spaced | Avoids head-only bias from metadata rows |

The sampling strategy takes the first 500 non-null values plus 500 evenly spaced values from the remainder. This avoids issues with:

- Header/metadata rows at the start
- Sparse data patterns
- Inconsistent data at file boundaries

### Fingerprint Sampling

| Setting | Value | Purpose |
|---------|-------|---------|
| Default sample | 100 rows | Content-based fingerprinting |
| Strategy | 50 head + 50 tail | Captures both ends of data |

### Full Data Operations

These operations process the **entire dataset** without sampling:

| Operation | Data Access | Notes |
|-----------|-------------|-------|
| File loading | All rows | Loaded into memory via pandas |
| Grain uniqueness | All rows | Tests actual row combinations |
| Null rate calculation | All rows | Accurate null statistics |
| Column statistics | All rows | Min, max, mean, cardinality |

## Memory Considerations

### Data Loading

Datasculpt loads the entire dataset into memory using pandas:

```python
# These operations load full data into memory
result = infer("large_file.csv")  # Loads via pd.read_csv
result = infer(existing_df)        # Uses existing DataFrame
```

Memory usage is reported in the `IntakeResult`:

```python
from datasculpt.intake import intake_file

result = intake_file("data.csv")
print(f"Memory: {result.preview.memory_usage_bytes / 1024 / 1024:.1f} MB")
```

### Memory Estimation

Rough memory estimates for common scenarios:

| Rows | Columns | Approximate Memory |
|------|---------|-------------------|
| 100K | 20 | 50-200 MB |
| 1M | 20 | 500 MB - 2 GB |
| 10M | 20 | 5-20 GB |

Actual memory depends on:
- Column data types (strings use more than integers)
- String lengths and cardinality
- Pandas internal optimizations

### Reducing Memory Usage

For large datasets, consider:

1. **Pre-filter rows** before calling `infer()`:
   ```python
   sample_df = large_df.sample(n=100_000)
   result = infer(sample_df)
   ```

2. **Load specific columns**:
   ```python
   df = pd.read_csv("large.csv", usecols=["col1", "col2", "col3"])
   result = infer(df)
   ```

3. **Use chunked loading** for very large files:
   ```python
   # Sample from a large file without loading it all
   chunks = pd.read_csv("huge.csv", chunksize=100_000)
   sample = next(chunks)  # First 100K rows
   result = infer(sample)
   ```

## Grain Inference Complexity

Grain inference tests column combinations for uniqueness. This has combinatorial complexity.

### Limits

| Setting | Default | Purpose |
|---------|---------|---------|
| `max_grain_columns` | 4 | Maximum columns in composite key |
| Candidate limit | 20 | Maximum columns to test combinations |
| Score threshold | 0.15 | Minimum score to consider candidate |

### Combinatorial Growth

Testing all combinations of N columns from C candidates:

| Candidates | Max Cols | Combinations Tested |
|------------|----------|---------------------|
| 10 | 4 | 385 |
| 15 | 4 | 1,940 |
| 20 | 4 | 6,195 |

Each combination requires a uniqueness check (groupby operation), so inference time grows with dataset size and candidate count.

### Performance Tuning

For faster inference on large datasets:

```python
from datasculpt.core.types import InferenceConfig

# Faster: fewer grain columns tested
fast_config = InferenceConfig(max_grain_columns=2)

# Slower but more thorough
thorough_config = InferenceConfig(max_grain_columns=5)
```

## Expected Behavior with Large Datasets

### Sub-second (< 1s)

- Small datasets (< 10K rows)
- Few columns (< 20)
- Simple grain (single column unique)

### 1-10 seconds

- Medium datasets (10K - 100K rows)
- Moderate columns (20-50)
- Composite grain search

### 10+ seconds

- Large datasets (> 100K rows)
- Many columns (> 50)
- Complex grain (3-4 column combinations)
- External adapters enabled (Frictionless, DataProfiler)

### Optimization Tips

1. **Disable unused adapters**:
   ```python
   config = InferenceConfig(
       use_frictionless=False,
       use_dataprofiler=False,
   )
   ```

2. **Reduce grain search space**:
   ```python
   config = InferenceConfig(max_grain_columns=2)
   ```

3. **Pre-sample large datasets**:
   ```python
   result = infer(df.sample(n=50_000))
   ```

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Via `pd.read_csv` |
| Excel | `.xlsx`, `.xls` | Via `pd.read_excel` |
| Parquet | `.parquet` | Via `pd.read_parquet` |
| Stata | `.dta` | Via `pd.read_stata` |

All formats are loaded entirely into memory. For very large files in Parquet or CSV format, consider pre-filtering with pandas before calling `infer()`.
