---
layout: default
title: Data Comparison & Accuracy Verification Roadmap
---

## Data Comparison & Accuracy Verification Roadmap

**Goal**: Ensure parser output can be accurately compared against verified ground truth data to maintain election data integrity.

**Status**: Planning Phase  
**Last Updated**: February 5, 2026

---

## 🎯 Executive Summary

### Current State

- ✅ Verification framework exists (DL1/DL2 architecture)
- ✅ Schema validation tests in place
- ✅ Confidence scoring implemented
- ❌ **No ground truth (DL1) reference dataset**
- ❌ **No automated comparison/diffing tools**
- ❌ **No accuracy regression detection**

### Critical Gap

**We have the infrastructure but no verified data to compare against.**

**Impact**: Cannot automatically verify parser accuracy, detect regressions, or quantify improvements.

---

## 📋 Implementation Phases

### Phase 1: Ground Truth Dataset Creation 🔴 CRITICAL

**Timeline**: Week 1-2  
**Priority**: HIGH  
**Effort**: Medium

#### 1.1 Create Verified Reference Dataset

**Action**: Build DL1 (verified ground truth) dataset with manually verified results.

**Steps**:

```bash
# Create DL1 directory structure
mkdir -p webapp/parser/fixtures/dl1/{html,pdf,json,csv}
mkdir -p webapp/parser/fixtures/dl2  # AI-extracted (unverified)
```

**Reference Data to Collect**:

| Format | Source | Verified By | Records |
| -------- | -------- | ------------- | --------- |
| HTML | 5 state election pages | Manual review | 50+ contests |
| PDF | Official county results | Manual review | 30+ contests |
| JSON | API exports | Cross-check | 40+ contests |
| CSV | Downloaded results | Manual review | 25+ contests |

**Verification Process**:

1. Human reviewer extracts data from source document
2. Second reviewer validates extraction
3. Store in `dl1/` with metadata:

   ```json
   {
     "dl1_id": "ca_sf_governor_2024",
     "source_url": "https://...",
     "verified_by": "reviewer@example.com",
     "verified_at": "2026-02-05T10:30:00Z",
     "contest": "Governor",
     "state": "California",
     "county": "San Francisco",
     "candidates": [
       {"name": "John Doe", "party": "Democratic", "votes": 123456, "percent": 55.2}
     ]
   }
   ```

#### 1.2 Document Verification Standards

**Create**: `docs/QUALITY/DL1_VERIFICATION_STANDARDS.md`

**Contents**:

- What constitutes "verified" data
- Double-entry verification process
- Acceptable error margins (e.g., ±0.1% for percentages due to rounding)
- Review/approval workflow
- Audit trail requirements

---

### Phase 2: Automated Comparison Tools 🟡 HIGH PRIORITY

**Timeline**: Week 2-3  
**Priority**: HIGH  
**Effort**: Medium-Large

#### 2.1 Build Comparison Engine

**Create**: `webapp/parser/utils/data_comparator.py`

**Features**:

```python
class DataComparator:
    """Compare parser output (DL2) against verified ground truth (DL1)."""
    
    def compare_datasets(
        self,
        dl1_data: dict,  # Ground truth
        dl2_data: dict,  # Parser output
        tolerance: dict = None  # Acceptable margins
    ) -> ComparisonResult:
        """
        Compare two datasets and return detailed diff report.
        
        Returns:
            - exact_matches: int (perfect matches)
            - near_matches: int (within tolerance)
            - mismatches: list[Difference]
            - missing_candidates: list[str]
            - extra_candidates: list[str]
            - vote_diff_summary: dict (avg/max vote differences)
        """
        pass
    
    def compute_accuracy_score(
        self,
        comparison_result: ComparisonResult
    ) -> float:
        """
        Compute overall accuracy score (0.0-1.0).
        
        Factors:
        - Candidate name match rate: 40%
        - Vote count accuracy: 30%
        - Party accuracy: 20%
        - Percent accuracy: 10%
        """
        pass
    
    def generate_diff_report(
        self,
        comparison_result: ComparisonResult,
        format: str = "markdown"
    ) -> str:
        """Generate human-readable comparison report."""
        pass
```

#### 2.2 Integrate with Test Suite

**Update**: `webapp/tests/test_data_accuracy.py` (NEW)

```python
import pytest
from webapp.parser.utils.data_comparator import DataComparator

class TestDataAccuracy:
    """Regression tests using DL1 ground truth."""
    
    @pytest.mark.parametrize("fixture_name", [
        "ca_sf_governor_2024_html",
        "tx_harris_senate_2024_pdf",
        "ny_nyc_mayor_2024_json",
    ])
    def test_parser_accuracy_vs_dl1(self, fixture_name):
        """Test parser output matches verified DL1 data."""
        # Load DL1 ground truth
        dl1_path = f"webapp/parser/fixtures/dl1/{fixture_name}.json"
        dl1_data = load_fixture(dl1_path)
        
        # Parse source document
        dl2_data = parse_document(dl1_data["source_document"])
        
        # Compare
        comparator = DataComparator()
        result = comparator.compare_datasets(dl1_data, dl2_data)
        
        # Assert accuracy threshold
        accuracy = comparator.compute_accuracy_score(result)
        assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} below 95% threshold"
        
        # Log detailed differences if below threshold
        if accuracy < 0.95:
            report = comparator.generate_diff_report(result)
            pytest.fail(f"Accuracy test failed:\n{report}")
```

---

### Phase 3: Regression Detection System 🟡 MEDIUM PRIORITY

**Timeline**: Week 3-4  
**Priority**: MEDIUM  
**Effort**: Medium

#### 3.1 Baseline Accuracy Metrics

**Create**: `webapp/parser/baselines/accuracy_baseline.json`

```json
{
  "baseline_version": "2026-02-05",
  "parser_version": "1.0.0",
  "overall_accuracy": 0.973,
  "by_format": {
    "html": 0.982,
    "pdf": 0.958,
    "json": 0.995,
    "csv": 0.971
  },
  "by_field": {
    "candidate_name": 0.991,
    "party": 0.987,
    "votes": 0.956,
    "percent": 0.944
  }
}
```

#### 3.2 Regression Detection Script

**Planned**: `scripts/detect_accuracy_regression.py` (not yet in repo; use this as a design sketch)

```python
#!/usr/bin/env python3
"""
Detect parser accuracy regressions by comparing current performance
against established baseline.
"""

def run_regression_tests():
    """Run all DL1 comparison tests and compare to baseline."""
    
    # 1. Run all DL1 accuracy tests
    test_results = run_pytest("webapp/tests/test_data_accuracy.py")
    
    # 2. Load baseline metrics
    baseline = load_json("webapp/parser/baselines/accuracy_baseline.json")
    
    # 3. Compare current vs baseline
    regressions = []
    for format_name, baseline_acc in baseline["by_format"].items():
        current_acc = test_results["by_format"][format_name]
        delta = current_acc - baseline_acc
        
        if delta < -0.02:  # More than 2% drop is regression
            regressions.append({
                "format": format_name,
                "baseline": baseline_acc,
                "current": current_acc,
                "delta": delta,
                "severity": "CRITICAL" if delta < -0.05 else "WARNING"
            })
    
    # 4. Generate report
    if regressions:
        print("🚨 ACCURACY REGRESSION DETECTED\n")
        for reg in regressions:
            print(f"  {reg['severity']}: {reg['format']}")
            print(f"    Baseline: {reg['baseline']:.2%}")
            print(f"    Current:  {reg['current']:.2%}")
            print(f"    Delta:    {reg['delta']:+.2%}\n")
        
        return 1  # Exit code 1 for CI failure
    else:
        print("✅ No accuracy regressions detected")
        return 0
```

#### 3.3 CI Integration

**Update**: `.github/workflows/main_ballotlens.yml`

Add regression check before deployment:

```yaml
- name: Run accuracy regression tests
  run: |
    # TODO: replace with regression script once implemented
    python -m pytest webapp/tests/test_data_accuracy.py
  continue-on-error: false  # Block deployment on regression
```

---

### Phase 4: Accuracy Metrics Dashboard 🟢 NICE-TO-HAVE

**Timeline**: Week 5-6  
**Priority**: LOW  
**Effort**: Medium

#### 4.1 Metrics Collection

**Extend**: `webapp/parser/quality_assurance/data_classifier.py`

Add accuracy tracking:

```python
def track_accuracy_metrics(dl1_id: str, dl2_result: dict, comparison: ComparisonResult):
    """Log accuracy metrics for dashboard visualization."""
    
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dl1_id": dl1_id,
        "accuracy_score": comparison.accuracy_score,
        "candidate_match_rate": comparison.candidate_match_rate,
        "vote_diff_avg": comparison.vote_diff_avg,
        "vote_diff_max": comparison.vote_diff_max,
        "format": dl2_result["format"],
        "state": dl2_result["state"],
        "contest": dl2_result["contest"]
    }
    
    # Append to metrics log
    with open("log/accuracy_metrics.jsonl", "a") as f:
        f.write(orjson.dumps(metrics).decode() + "\n")
```

#### 4.2 Dashboard Visualization

**Update**: `webapp/templates/quality_dashboard.html`

Add accuracy tracking section showing:

- **Overall accuracy trend** (line chart over time)
- **Accuracy by format** (bar chart: HTML vs PDF vs JSON vs CSV)
- **Field-level accuracy** (heatmap: candidate name, party, votes, percent)
- **Recent regressions** (table with details)
- **DL1 coverage** (% of parser output verified against ground truth)

---

## 🛠️ Quick Start Guide

### Step 1: Create First DL1 Reference

```bash
# 1. Create directory
mkdir -p webapp/parser/fixtures/dl1/html

# 2. Manually verify a simple HTML election result
# Open source: https://example.com/election-2024.html
# Extract data carefully into JSON:

cat > webapp/parser/fixtures/dl1/html/ca_sf_governor_2024.json << 'EOF'
{
  "dl1_id": "ca_sf_governor_2024",
  "source_url": "https://sfelections.sfgov.org/november-5-2024...",
  "source_hash": "sha256:abc123...",
  "verified_by": "reviewer@example.com",
  "verified_at": "2026-02-05T14:30:00Z",
  "review_method": "manual_double_entry",
  "contest": "Governor",
  "state": "California",
  "county": "San Francisco",
  "election_date": "2024-11-05",
  "candidates": [
    {
      "name": "Gavin Newsom",
      "party": "Democratic",
      "votes": 245678,
      "percent": 67.3,
      "winner": true
    },
    {
      "name": "Brian Dahle",
      "party": "Republican",
      "votes": 119234,
      "percent": 32.7,
      "winner": false
    }
  ],
  "total_votes": 364912,
  "verification_notes": "Data extracted from official county website. Vote totals match PDF Statement of Vote."
}
EOF
```

### Step 2: Run First Comparison Test

```bash
# Create simple comparison test (manual for now)
python << 'PYEOF'
import orjson
from webapp.parser.html_election_parser import parse_html

# Load DL1 ground truth
with open("webapp/parser/fixtures/dl1/html/ca_sf_governor_2024.json", "rb") as f:
    dl1 = orjson.loads(f.read())

# Parse the same source document
dl2 = parse_html(dl1["source_url"])

# Manual comparison
print("DL1 (ground truth):", dl1["candidates"])
print("DL2 (parser output):", dl2["candidates"])

# Count matches
matches = 0
for dl1_cand in dl1["candidates"]:
    for dl2_cand in dl2["candidates"]:
        if (dl1_cand["name"] == dl2_cand["name"] and
            abs(dl1_cand["votes"] - dl2_cand["votes"]) <= 5):  # Allow ±5 votes
            matches += 1
            break

accuracy = matches / len(dl1["candidates"])
print(f"\nAccuracy: {accuracy:.1%} ({matches}/{len(dl1['candidates'])} candidates matched)")
PYEOF
```

### Step 3: Expand DL1 Coverage

Target coverage goals:

- **Phase 1**: 10 verified fixtures (2 per format)
- **Phase 2**: 50 verified fixtures (diverse states/contests)
- **Phase 3**: 100+ verified fixtures (comprehensive coverage)

---

## 🎯 Success Metrics

| Metric | Target | Current | Status |
| -------- | -------- | --------- | --------- |
| DL1 fixtures created | 50+ | 0 | ❌ Not started |
| Automated comparison tests | 20+ | 0 | ❌ Not started |
| Overall parser accuracy | ≥95% | Unknown | ❌ Not measured |
| Regression detection | Automated | Manual | ❌ No automation |
| Accuracy dashboard | Live | None | ❌ Not built |

---

## 📚 Related Documents

- [Verification Framework](../QUALITY/VERIFICATION.md) - Quality assurance overview
- [Schema Validation Tests](../../webapp/tests/test_schema_validation.py) - Existing tests
- [DL1/DL2 Architecture](../../webapp/parser/utils/verification_framework.py) - Framework code
- [Local Sync Implementation](../../webapp/parser/verification/local_dl_sync.py) - DL1/DL2 sync

---

## 🔄 Next Actions

### Immediate (This Week)

1. ✅ Document current state (this file)
2. ⬜ Create DL1 verification standards doc
3. ⬜ Build first 5 DL1 reference datasets (1 per format + 1 complex)
4. ⬜ Write basic data comparison utility

### Short-term (Next 2 Weeks)

1. ⬜ Create `DataComparator` class with diff logic
2. ⬜ Add first automated accuracy test
3. ⬜ Baseline current parser accuracy

### Medium-term (Next Month)

1. ⬜ Expand DL1 coverage to 20+ fixtures
2. ⬜ Build regression detection script
3. ⬜ Integrate accuracy checks into CI/CD
4. ⬜ Add accuracy tracking to quality dashboard

---

## 💡 Open Questions

1. **Who will perform DL1 verification?**
   - Option A: You (manual review of source documents)
   - Option B: Automated cross-check against official APIs (where available)
   - Option C: Community verification model (multiple reviewers)

2. **What accuracy threshold is acceptable?**
   - Candidate names: 99%+? (critical for democracy)
   - Vote counts: 95%+? (some rounding acceptable)
   - Percentages: 90%+? (varies by calculation method)

3. **How to handle edge cases?**
   - Write-in candidates
   - Tied races
   - Provisional/overseas ballots
   - Recounts

4. **Storage strategy for DL1?**
   - Git LFS for large datasets?
   - Separate DL1 repository?
   - Cloud storage with hashing?

---

**Last Updated**: February 5, 2026  
**Maintained By**: Development Team  
**Review Cycle**: Bi-weekly during active development
