# Suggested Camelot noise overrides

Min count cutoff: 1

## State-level additions

```python
CAMELOT_STATE_NOISE_OVERRIDES.update({
    "florida": {
        "boilerplate": [
            r"^\s*Precincts\ Reporting:\ 10\ /\ 10\s*$",
        ],
    },
    "new_york": {
        "row": [
            r"^\s*Party=Blank/Undervote\s*$",
        ],
    },
})
```

## County-level additions

```python
CAMELOT_COUNTY_NOISE_OVERRIDES.update({
    ("new_york", "new york"): {
        "title": [
            r"^\s*Statement\ and\ Return\ Report\s*$",
        ],
    },
})
```
