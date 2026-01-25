# Ambiguous Shape

When Datasculpt can't confidently distinguish between shapes.

## The Problem

Some datasets genuinely fit multiple interpretations. Consider:

```csv
region,category,count,total,rate
North,A,100,1000,0.10
North,B,150,1000,0.15
South,A,80,800,0.10
South,B,120,800,0.15
```

Is this:
- **Wide observations** with three measures (`count`, `total`, `rate`)?
- **Long indicators** if `category` represents different metrics?

Both are defensible. The data structure alone doesn't tell us.

## Detecting Ambiguity

```python
from datasculpt import infer

result = infer("ambiguous.csv", interactive=True)

>>> result.decision_record.hypotheses[:2]
[
    HypothesisScore(
        hypothesis=<ShapeHypothesis.WIDE_OBSERVATIONS>,
        score=0.58,
        reasons=['Multiple numeric columns suggest measures']
    ),
    HypothesisScore(
        hypothesis=<ShapeHypothesis.LONG_INDICATORS>,
        score=0.52,
        reasons=['category column has indicator-like values']
    )
]
```

The gap between the top two scores is only 0.06 — below the default threshold of 0.10.

```python
>>> result.decision_record.hypotheses[0].score - result.decision_record.hypotheses[1].score
0.06

>>> from datasculpt.core.types import InferenceConfig
>>> InferenceConfig().hypothesis_confidence_gap
0.1
```

## Generated Questions

When ambiguity is detected in interactive mode, Datasculpt generates questions:

```python
>>> result.pending_questions
[
    Question(
        id='q_abc12345',
        type=<QuestionType.CHOOSE_ONE>,
        prompt='The dataset shape is ambiguous. Please select the most appropriate shape:',
        choices=[
            {'value': 'wide_observations', 'label': 'Wide Observations', 'score': 0.58},
            {'value': 'long_indicators', 'label': 'Long Indicators', 'score': 0.52},
            {'value': 'long_observations', 'label': 'Long Observations', 'score': 0.45}
        ],
        default='wide_observations',
        rationale='Score gap between top hypotheses is 0.06, below threshold 0.10'
    )
]
```

## Resolving Ambiguity

Provide an answer to resolve:

```python
from datasculpt import apply_answers

answers = {result.pending_questions[0].id: "long_indicators"}
result = apply_answers(result, answers)

>>> result.proposal.shape_hypothesis
<ShapeHypothesis.LONG_INDICATORS: 'long_indicators'>

>>> result.pending_questions
[]  # No more questions
```

## Multiple Ambiguities

A dataset can have multiple ambiguous aspects:

```python
>>> result.pending_questions
[
    Question(
        id='q_shape_123',
        prompt='The dataset shape is ambiguous...',
        ...
    ),
    Question(
        id='q_grain_456',
        prompt='Please confirm or select the grain...',
        ...
    ),
    Question(
        id='q_role_789',
        prompt='What is the role of column "category"?',
        ...
    )
]
```

Provide all answers at once:

```python
answers = {
    'q_shape_123': 'long_indicators',
    'q_grain_456': ['region', 'category'],
    'q_role_789': 'indicator_name'
}
result = apply_answers(result, answers)
```

## Tuning Sensitivity

Adjust the confidence gap threshold to be more or less sensitive:

```python
from datasculpt.core.types import InferenceConfig

# More sensitive (more questions)
config = InferenceConfig(hypothesis_confidence_gap=0.15)

# Less sensitive (fewer questions)
config = InferenceConfig(hypothesis_confidence_gap=0.05)

result = infer("data.csv", config=config, interactive=True)
```

## Non-Interactive Mode

Without `interactive=True`, Datasculpt picks the top-scoring hypothesis but records the ambiguity:

```python
result = infer("ambiguous.csv")  # No interactive flag

>>> result.proposal.shape_hypothesis
<ShapeHypothesis.WIDE_OBSERVATIONS: 'wide_observations'>

>>> result.proposal.warnings
['Shape detection confidence is low (gap: 0.06). Consider manual review.']
```

## When Ambiguity Is Expected

Some datasets are genuinely ambiguous without domain context:

| Scenario | Why Ambiguous |
|----------|---------------|
| Survey data | Rows could be responses or pivoted metrics |
| Aggregated reports | Multiple interpretations of granularity |
| ETL staging tables | Intermediate format, not yet normalized |
| API exports | Format depends on client expectations |

In these cases, interactive mode is the right approach — let domain experts resolve ambiguity.

## Decision Record

Even when ambiguous, the decision record captures the full analysis:

```python
>>> record = result.decision_record
>>> record.selected_hypothesis
<ShapeHypothesis.WIDE_OBSERVATIONS: 'wide_observations'>

>>> record.hypotheses
[
    HypothesisScore(hypothesis=WIDE_OBSERVATIONS, score=0.58, ...),
    HypothesisScore(hypothesis=LONG_INDICATORS, score=0.52, ...),
    HypothesisScore(hypothesis=LONG_OBSERVATIONS, score=0.45, ...),
    ...
]

>>> record.answers
{'q_abc12345': 'wide_observations'}  # If resolved via answer
```

## See Also

- [Grain Detection](grain-detection.md) — Ambiguity in unique key detection
- [Mental Model](../getting-started/mental-model.md) — The evidence → hypothesis → decision pipeline
