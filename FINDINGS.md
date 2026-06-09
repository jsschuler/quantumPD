# Experimental Findings Log

## Experiment: Network Prisoner's Dilemma — Classical vs Quantum Annealing

**Date:** 2026-06-09
**Parameters:** N=10, R=3, S=0, T=5, P=1, 50 replications per parameter point, 100 SA runs, 100 SQA runs per instance.

---

## Round 1 — Initial Run (pre-fix)

### Erdős-Rényi sweep

| p | edges (mean) | spectral gap | SA coincidence | SA approx | SQA coincidence | SQA approx | EQ approx |
|---|---|---|---|---|---|---|---|
| 0.2 | 8.7 | 0.041 | 0.515 | 1.000 | 0.000 | 0.333 | 1.000 |
| 0.4 | 17.1 | 0.351 | 0.953 | 1.000 | 0.006 | 0.342 | 1.000 |
| 0.6 | 26.9 | 0.638 | 0.997 | 0.998 | 0.144 | 0.625 | 1.000 |
| 0.8 | 36.0 | 0.824 | 0.987 | 0.991 | 0.762 | 0.888 | 1.000 |
| 1.0 | 45.0 | 1.111 | 0.970 | 0.980 | 0.653 | 0.769 | 1.000 |

### Watts-Strogatz sweep

| beta | edges | spectral gap | clustering | SA coincidence | SA approx | SQA coincidence | SQA approx | EQ approx |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 20.0 | 0.441 | 0.500 | 1.000 | 1.000 | 0.006 | 0.342 | 1.000 |
| 0.25 | 20.0 | 0.433 | 0.403 | 1.000 | 1.000 | 0.000 | 0.334 | 1.000 |
| 0.5 | 20.0 | 0.484 | 0.376 | 1.000 | 1.000 | 0.000 | 0.334 | 1.000 |
| 0.75 | 20.0 | 0.491 | 0.380 | 1.000 | 1.000 | 0.000 | 0.334 | 1.000 |
| 1.0 | 20.0 | 0.503 | 0.374 | 1.000 | 1.000 | 0.000 | 0.333 | 1.000 |

### Structural extremes

| Graph | Edges | Spectral gap | Clustering | Frustration | SA coincidence | SA approx | SQA coincidence | SQA approx | EQ approx |
|---|---|---|---|---|---|---|---|---|---|
| Complete K₁₀ | 45 | 1.111 | 1.0 | 1.0 | 0.97 | 0.980 | 0.66 | 0.773 | 1.000 |
| Bipartite K₅,₅ | 25 | 1.000 | 0.0 | 0.0 | 1.00 | 1.000 | 0.42 | 0.698 | 1.000 |

### Observations (Round 1)

- **EQ is perfect everywhere.** Approx ratio 1.000 and ground-state overlap 1.000 on every instance across all topologies. The adiabatic gap never closes at n=10; the problem is uniformly easy for exact quantum annealing.
- **SA is strong.** Coincidence rate near 1.0 for p≥0.4 (ER) and all beta (WS). The only weakness is sparse ER (p=0.2) where it still achieves approx ratio 1.000 despite not always finding the exact configuration (implying multiple optimal configurations).
- **SQA appears broken.** Approx ratio locked at ~0.333 across all WS instances and sparse ER. This equals P·|E| / (R·|E|) = 1/3, the normalized welfare of the all-defect configuration. SQA was converging to all-defect (cooperation rate ≈ 0) in nearly every run.
- **SQA variability in ER tracked edge count, not topology.** WS holds edge count fixed at 20 and shows zero variation with rewiring. ER improves monotonically with p only because p increases both density and edge count simultaneously. The improvement was about energy scale, not structure.
- **Non-monotonicity at p=1.0 (ER).** Complete graph (p=1.0) was harder for SQA than p=0.8 despite having more edges, likely because maximal frustration (frustration=1.0) creates a more complex energy landscape.

---

## Bug Fix — SQA best-solution tracking

**File:** `src/sqa.jl`

**Bug:** The `run_sqa` function initialized `best_welfare = -Inf` and `best_config` but only updated them in a final loop over Trotter slices *after* the MC loop completed. Any good configuration visited during the run and subsequently abandoned was lost. The algorithm effectively reported the final state, not the best state ever seen.

**Fix:** Update `best_welfare` and `best_config` inside the MC loop immediately after an accepted flip, using the already-computed `w_new`.

```julia
# Before (only at end of run):
for tau in 1:P
    w = welfare_from_config(replicas[tau], Q, offset)
    if w > best_welfare ...

# After (inside MC loop, on acceptance):
if delta_total <= 0.0 || rand(rng) < exp(-beta * delta_total)
    replicas[tau][i] = new_val
    if w_new > best_welfare
        best_welfare = w_new
        best_config = copy(replicas[tau])
    end
end
```

---

## Round 2 — Post-fix Run

### Erdős-Rényi sweep

| p | SQA coincidence | SQA approx | (prev approx) |
|---|---|---|---|
| 0.2 | 0.357 | 0.986 | 0.333 |
| 0.4 | 0.943 | 1.000 | 0.342 |
| 0.6 | 1.000 | 1.000 | 0.625 |
| 0.8 | 1.000 | 1.000 | 0.888 |
| 1.0 | 1.000 | 1.000 | 0.769 |

### Watts-Strogatz sweep

All beta values: SQA coincidence = 1.000, approx = 1.000. Previously all stuck at ~0.333.

### Structural extremes

| Graph | SQA coincidence | SQA approx | (prev approx) |
|---|---|---|---|
| Complete K₁₀ | 1.000 | 1.000 | 0.773 |
| Bipartite K₅,₅ | 1.000 | 1.000 | 0.698 |

### Observations (Round 2)

- The single bug fix resolved essentially all of SQA's poor performance. The algorithm was finding optimal configurations during runs but discarding them.
- **Remaining weakness:** ER p=0.2 — SQA coincidence 0.357, approx 0.986. Sparse graphs with many disconnected components (mean spectral gap 0.041) remain harder. SQA finds near-optimal solutions but misses the exact optimum more often. This is the one regime where SA (coincidence 0.515) still outperforms SQA slightly on coincidence rate, though both achieve approx ratio ≈ 1.0.
- **WS insensitivity confirmed.** With the fix, SQA is perfect across all WS topologies. This confirms the earlier conjecture: rewiring probability (and the small-world transition) does not affect solution quality at n=10. Neither SA nor SQA struggles with WS graphs at any beta.
- **Implication for the main conjecture.** After the fix, both SA and SQA find the optimum reliably across all tested topologies except sparse ER. There is no regime at n=10 where SQA outperforms SA in a meaningful way — both are near-ceiling. The conjecture that quantum annealing provides a structural advantage on frustrated intermediate-density networks cannot be confirmed or falsified at this scale; the problem is too easy for both methods. Testing the conjecture likely requires larger n, where the energy landscape becomes genuinely hard.

---

## Open questions / next steps

- **Why does p=0.2 remain hard for both methods?** Many p=0.2 graphs are disconnected (spectral gap=0); the optimal configuration may be less constrained, making exact-match coincidence a weaker metric. Investigate whether the "missed" instances are genuinely suboptimal or just alternative optima with equal welfare.
- **SQA parameter sensitivity.** The fix resolves the bug but issues #2–#4 (step count per DOF, energy-scale mismatch at low density, J_perp guard at gamma=0) remain unaddressed. Worth revisiting if p=0.2 performance matters.
- **Scale-up.** To test the main conjecture, repeat at n=15 or n=20. Exact solver becomes infeasible above n~20; EQ becomes infeasible above n~15. The interesting regime is where SA starts failing and SQA may or may not follow.
