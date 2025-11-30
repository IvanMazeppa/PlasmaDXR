# Critical Bug Fix: Accuracy Score Calculation

## Date: 2025-11-30
## Impact: **HIGH** - All previous GA optimization results need revalidation

---

## Bug Description

The `accuracy_score` was always returning **0/100** regardless of actual Keplerian accuracy, causing:
- Overall scores ~40-50 instead of 80-90
- GA optimizer couldn't differentiate based on physics accuracy
- "UNSUITABLE" recommendations for actually good configurations

## Root Cause

In `BenchmarkMetrics.h`, the `ComputeScores()` function used `accuracy.radialForceSignCorrectPercent` before it was computed:

```cpp
// BenchmarkRunner.cpp - ComputeFinalMetrics()
void BenchmarkRunner::ComputeFinalMetrics() {
    m_results.Finalize();  // <-- ComputeScores() called here
    
    // radialForceSignCorrectPercent set AFTER Finalize()!
    m_results.accuracy.radialForceSignCorrectPercent = 
        (m_results.accuracy.avgRadialForce.mean < 0.0f) ? 100.0f : 0.0f;
}
```

Since `radialForceSignCorrectPercent` was still 0.0f when scores were computed:
```cpp
accuracyScore -= (100.0f - accuracy.radialForceSignCorrectPercent);
// = accuracyScore -= (100.0f - 0.0f)
// = accuracyScore -= 100  <-- OOPS!
```

## The Fix

Moved `radialForceSignCorrectPercent` computation INTO `ComputeScores()`, which runs AFTER `accuracy.Finalize()` provides `avgRadialForce.mean`:

```cpp
// BenchmarkMetrics.h - ComputeScores()
void ComputeScores() {
    // ... stability score ...
    
    // Compute radial force sign correctness (must be here!)
    accuracy.radialForceSignCorrectPercent = 
        (accuracy.avgRadialForce.mean < 0.0f) ? 100.0f : 0.0f;
    
    // Accuracy score (0-100)
    accuracyScore = 100.0f;
    accuracyScore -= accuracy.keplerianVelocityError.mean * 5.0f;
    accuracyScore -= (100.0f - accuracy.radialForceSignCorrectPercent);
    accuracyScore = std::clamp(accuracyScore, 0.0f, 100.0f);
    
    // ... rest of scores ...
}
```

## Verification Results

| Metric | Before Fix | After Fix | Change |
|--------|------------|-----------|--------|
| **Accuracy** | 0/100 | **95.0/100** | +95 pts |
| **Stability** | 40.5/100 | **73.3/100** | +32.8 pts |
| **Performance** | 100/100 | 100/100 | - |
| **Visual** | 47.1/100 | 48.9/100 | +1.8 pts |
| **OVERALL** | 41.25/100 | **81.5/100** | **+40.3 pts** |
| Recommendation | UNSUITABLE | **GOOD** | ✅ |

## Files Modified

- `src/benchmark/BenchmarkMetrics.h` (line ~217)
- `src/benchmark/BenchmarkRunner.cpp` (line ~445)

## Impact on Previous Results

**All previous GA optimization results are now INVALID** because:
1. Fitness was computed with accuracy=0 always
2. GA couldn't optimize for physics accuracy
3. Overall scores were ~40 pts lower than reality

**The best individual (fitness 47.20) would now score ~85-90** with this fix.

## Recommended Actions

1. ✅ Bug is fixed and verified
2. ⏳ Re-run GA optimization with corrected scoring
3. ⏳ Previous "best" parameters may not be optimal for accuracy
4. ⏳ Update FINAL_BENCHMARK_RESULTS.md with corrected numbers

## Lessons Learned

1. **Initialization order matters** - compute dependencies BEFORE using them
2. **Zero values are suspicious** - accuracy=0 despite 1.8% error was a red flag
3. **Verify scoring formula manually** - trace through calculation with real numbers

