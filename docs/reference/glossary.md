# Glossary

Terminology used in Datasculpt.

## A

### Ambiguity

When Datasculpt cannot confidently distinguish between two or more interpretations. Measured by the score gap between hypotheses. Triggers questions in interactive mode.

### Array Profile

Statistics about array-type columns: average length, min/max length, consistency.

## C

### Cardinality

The number of distinct values in a column. High cardinality suggests uniqueness (keys, measures). Low cardinality suggests categories (dimensions).

### Column Evidence

See [Evidence](#evidence).

### Confidence

A score from 0.0 to 1.0 indicating certainty in an inference. Low confidence triggers warnings or questions.

## D

### Decision Record

Complete audit trail for an inference run. Contains selected hypothesis, alternatives, evidence, questions, and answers.

### Dimension

A column role for categorical grouping variables. Low cardinality, used in GROUP BY clauses.

### Distinct Ratio

The fraction of unique values in a column. Calculated as `unique_count / row_count`.

## E

### Evidence

Objective facts about a column: type, cardinality, null rate, value distribution. Separated from interpretation.

## G

### Grain

The minimal set of columns that uniquely identifies each row. Also called the "unique key" or "natural key".

### Grain Diagnostics

Details about grain quality: duplicate count, null count, example duplicates.

## H

### Hypothesis

A candidate interpretation. Shape hypotheses are the five structural patterns. Hypotheses are scored and ranked.

### Hypothesis Score

A score from 0.0 to 1.0 for a shape hypothesis, with supporting reasons.

## I

### Indicator

In statistical data, a named metric (e.g., "population", "gdp"). In long_indicators shape, stored as indicator_name/value pairs.

### Inference

The process of determining structure from data. Datasculpt infers shape, grain, and roles.

### Interactive Mode

Mode where Datasculpt generates questions for ambiguous aspects instead of guessing.

### Invariant Proposal

Output ready for registration with Invariant. Contains shape, grain, columns, warnings.

## K

### Key

A column role for uniqueness contributors. Part of the grain.

## L

### Long Indicators

Dataset shape where metrics are stored as indicator_name/value pairs, one per row.

### Long Observations

Dataset shape where each row is an atomic observation with dimensions and measures as columns.

## M

### Measure

A column role for numeric, aggregatable values. High cardinality, used in SUM/AVG/COUNT.

### Metadata

A column role for descriptive, non-analytical columns. Notes, comments, labels.

## N

### Null Rate

The fraction of missing values in a column. High null rates suggest optional or metadata columns.

## P

### Primitive Type

Basic data type: string, integer, number, boolean, date, datetime.

### Pseudo-Key

A column that appears unique but doesn't represent a meaningful business key. Examples: row_id, uuid, created_at.

## Q

### Question

In interactive mode, a prompt for user input to resolve ambiguity. Types: choose_one, choose_many, confirm.

## R

### Role

The structural purpose of a column: key, dimension, measure, time, indicator_name, value, series, metadata.

### Role Score

A likelihood score (0.0 to 1.0) for each possible role assignment.

## S

### Series

A column role for embedded time series stored as arrays or objects.

### Shape

The structural pattern of a dataset: long_observations, long_indicators, wide_observations, wide_time_columns, series_column.

### Shape Result

The output of shape detection: selected hypothesis, ranked alternatives, ambiguity status.

### Structural Metadata

Metadata about layout and intent: shape, grain, roles. Distinct from technical, business, operational, or governance metadata.

### Structural Type

How values are structured: scalar, array, object.

## T

### Time

A column role for temporal dimensions. Date or datetime type.

## U

### Uniqueness Ratio

The fraction of rows with unique grain values. 1.0 means no duplicates.

## V

### Value

A column role in unpivoted data that holds the numeric value paired with an indicator name.

### Value Profile

Distribution statistics for numeric columns: min, max, mean, and ratios for bounded ranges.

## W

### Wide Observations

Dataset shape with measures as columns. Spreadsheet-style layout.

### Wide Time Columns

Dataset shape with time periods encoded in column headers (e.g., 2022, 2023, 2024).
