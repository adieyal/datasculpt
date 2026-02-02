# Microdata

Survey and observation data with coded question columns, hierarchical IDs, and geography levels.

## The Data

```csv
hhid,indiv,zone,state,lga,weight,s1aq1,s1aq2,v101,v102
1001,1,North Central,Kogi,Dekina,1.234,1,35,2,3
1001,2,North Central,Kogi,Dekina,1.234,2,28,2,1
1002,1,South West,Lagos,Ikeja,0.987,2,42,1,2
1002,2,South West,Lagos,Ikeja,0.987,1,19,1,4
1003,1,North Central,Kwara,Ilorin,1.456,1,55,3,2
```

## What It Looks Like

| hhid | indiv | zone | state | lga | weight | s1aq1 | s1aq2 | v101 | v102 |
|------|-------|------|-------|-----|--------|-------|-------|------|------|
| 1001 | 1 | North Central | Kogi | Dekina | 1.234 | 1 | 35 | 2 | 3 |
| 1001 | 2 | North Central | Kogi | Dekina | 1.234 | 2 | 28 | 2 | 1 |
| 1002 | 1 | South West | Lagos | Ikeja | 0.987 | 2 | 42 | 1 | 2 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## The Inference

```python
from datasculpt import infer

result = infer("household_survey.csv")
```

### Shape Detection

```python
>>> result.proposal.shape_hypothesis
<ShapeHypothesis.MICRODATA: 'microdata'>

>>> result.decision_record.hypotheses[0]
HypothesisScore(
    hypothesis=<ShapeHypothesis.MICRODATA>,
    score=0.75,
    reasons=[
        'Found respondent ID column(s): hhid',
        'Found subunit ID column(s): indiv',
        'Found geography hierarchy (3 levels): zone, state, lga',
        'Found survey weight column: weight',
        'Moderate ratio (40%) of coded question columns'
    ]
)
```

### Grain Detection

```python
>>> result.decision_record.grain
GrainInference(
    key_columns=['hhid', 'indiv'],
    confidence=0.95,
    uniqueness_ratio=1.0,
    evidence=[
        'Combination of hhid, indiv is unique',
        'Hierarchical ID structure: primary (hhid) + subunit (indiv)'
    ]
)
```

### Role Assignments

| Column | Role | Evidence |
|--------|------|----------|
| hhid | respondent_id | Matches household ID pattern (hhid) |
| indiv | subunit_id | Matches individual/member pattern |
| zone | geography_level | Geographic administrative level |
| state | geography_level | Geographic administrative level |
| lga | geography_level | Local Government Area pattern |
| weight | survey_weight | Matches weight column pattern |
| s1aq1 | question_response | LSMS section 1a question pattern |
| s1aq2 | question_response | LSMS section 1a question pattern |
| v101 | question_response | DHS variable pattern |
| v102 | question_response | DHS variable pattern |

## Why This Shape

Datasculpt detected `microdata` because:

1. **Hierarchical ID structure** - `hhid` identifies households, `indiv` identifies individuals within households
2. **Coded question columns** - `s1aq1`, `s1aq2` follow LSMS patterns (section 1, part a, question 1/2); `v101`, `v102` follow DHS patterns
3. **Geography hierarchy** - `zone`, `state`, `lga` form an ordered geographic hierarchy (broad to narrow)
4. **Survey weight present** - `weight` column indicates complex survey design
5. **No indicator/value pattern** - Unlike long_indicators, values are in separate columns per question

## Detection Signals

The microdata detector looks for these patterns:

| Signal | Example Patterns | Weight |
|--------|------------------|--------|
| Respondent ID | hhid, hh_id, household_id, personid | +0.15 |
| Subunit ID | indiv, member_num, child_num | +0.10 |
| Cluster ID | ea, cluster, psu | +0.05 |
| Geography levels | zone, state, region, lga, district | +0.10 |
| Survey weight | weight, wgt, hh_weight | +0.05 |
| Question columns | s1aq1, v101, hv001, q1 | +0.30 (if >50%) |
| High column count | 50+ columns | +0.25 |

## Survey Type Inference

Datasculpt can infer the survey type from question column patterns:

```python
>>> from datasculpt.core.microdata import infer_survey_type

>>> infer_survey_type("s1aq1")
'LSMS'

>>> infer_survey_type("hv001")
'DHS'

>>> infer_survey_type("v101")
'DHS'

>>> infer_survey_type("wm1")
'MICS'
```

## The Proposal

```python
>>> result.proposal
InvariantProposal(
    dataset_name='household_survey',
    dataset_kind=<DatasetKind.MICRODATA>,
    shape_hypothesis=<ShapeHypothesis.MICRODATA>,
    grain=['hhid', 'indiv'],
    columns=[
        ColumnSpec(name='hhid', role=<Role.RESPONDENT_ID>, ...),
        ColumnSpec(name='indiv', role=<Role.SUBUNIT_ID>, ...),
        ColumnSpec(name='zone', role=<Role.GEOGRAPHY_LEVEL>, ...),
        ColumnSpec(name='state', role=<Role.GEOGRAPHY_LEVEL>, ...),
        ColumnSpec(name='lga', role=<Role.GEOGRAPHY_LEVEL>, ...),
        ColumnSpec(name='weight', role=<Role.SURVEY_WEIGHT>, ...),
        ColumnSpec(name='s1aq1', role=<Role.QUESTION_RESPONSE>, ...),
        ColumnSpec(name='s1aq2', role=<Role.QUESTION_RESPONSE>, ...),
        ColumnSpec(name='v101', role=<Role.QUESTION_RESPONSE>, ...),
        ColumnSpec(name='v102', role=<Role.QUESTION_RESPONSE>, ...),
    ],
    warnings=[],
    required_user_confirmations=[]
)
```

## See Also

- [Wide Observations](wide-observations.md) - Standard wide format without survey patterns
- [Long Indicators](long-indicators.md) - Unpivoted indicator/value format
