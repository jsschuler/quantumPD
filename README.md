# Network Prisoner's Dilemma: Classical vs Quantum Annealing

A computational study comparing classical simulated annealing (SA), simulated quantum annealing (SQA), and exact adiabatic quantum annealing on a welfare-maximization problem over network prisoner's dilemma games. The central question is whether quantum annealing finds better welfare optima than classical SA, and whether any advantage depends on graph structure.

## Research question

Given a graph $G = (V, E)$ where each edge hosts a prisoner's dilemma, agents choose to cooperate or defect to maximize total pairwise welfare. This is cast as a QUBO problem and solved by three methods across two random graph models (Erdős-Rényi and Watts-Strogatz) and two structural extremes (complete graph $K_{10}$, balanced bipartite $K_{5,5}$).

## Methods

| Method | Description |
|---|---|
| **SA** | Simulated annealing with geometric cooling, single spin-flip proposals |
| **SQA** | Path-integral Monte Carlo (Trotter decomposition), $P=20$ slices, transverse field annealed from $\Gamma_0=3$ to $0$ |
| **EQ** | Exact adiabatic evolution via diagonalization of the full $2^n \times 2^n$ Hamiltonian |
| **Exact** | Brute-force enumeration of all $2^{10} = 1024$ configurations (ground truth) |

Quantum simulation is implemented from scratch using `LinearAlgebra` — no quantum computing SDKs.

## Key results (n = 10)

**After fixing a best-solution tracking bug in SQA**, all three methods find the welfare optimum reliably across most topologies:

- **Watts-Strogatz:** SA and SQA both achieve coincidence rate 1.0 across all rewiring probabilities $\beta \in [0, 1]$. The small-world transition has no effect on solution quality at this scale.
- **Structural extremes:** SA and SQA both find the optimum on $K_{10}$ and $K_{5,5}$.
- **Erdős-Rényi (sparse, $p=0.2$):** The only regime where either method struggles. Both SA and SQA find near-optimal solutions (approx ratio ≈ 0.99) but exact coincidence is below 1.0, likely due to multiple near-degenerate optima in sparse disconnected graphs.
- **Exact quantum:** Perfect across all instances (approx ratio 1.0, ground-state overlap 1.0 everywhere).

The main conjecture — that SQA outperforms SA on intermediate-density frustrated networks — cannot be confirmed or refuted at $n=10$: the problem is too easy for both methods at this scale. Larger $n$ is needed to open a meaningful hardness gap.

## Payoff structure

```
R = 3  (mutual cooperation)
S = 0  (sucker payoff)
T = 5  (temptation to defect)
P = 1  (mutual defection)
```

Satisfies PD conditions: $T > R > P > S$ and $2R > T + S$.

## Experimental parameters

```julia
N_NODES        = 10
N_REPLICATIONS = 50   # random graph instances per parameter point
N_SA_RUNS      = 100  # SA runs per instance
N_SQA_RUNS     = 100  # SQA runs per instance

ER_P_VALUES    = [0.2, 0.4, 0.6, 0.8, 1.0]
WS_K           = 4
WS_BETA        = [0.0, 0.25, 0.5, 0.75, 1.0]
```

## Repository structure

```
├── src/
│   ├── QuantumPD.jl        # module entrypoint
│   ├── graphs.jl           # graph generation and structural metrics
│   ├── qubo.jl             # QUBO encoding of network PD welfare
│   ├── exact_solver.jl     # brute-force over all 2^n configurations
│   ├── classical_sa.jl     # simulated annealing
│   ├── sqa.jl              # simulated quantum annealing (SQA)
│   ├── exact_quantum.jl    # exact Schrödinger evolution
│   └── metrics.jl          # approximation ratio, logical entropy, coincidence
├── experiments/
│   ├── run_erdos_renyi.jl
│   ├── run_watts_strogatz.jl
│   └── run_extremes.jl
├── plots/
│   └── heatmaps.jl         # generates all figures from CSVs
├── results/                # CSV outputs (one file per experiment)
├── test/
│   └── runtests.jl
└── FINDINGS.md             # detailed results log
```

## Running

Requires Julia 1.10+.

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. test/runtests.jl

# Run experiments (in order)
julia --threads auto --project=. experiments/run_erdos_renyi.jl
julia --threads auto --project=. experiments/run_watts_strogatz.jl
julia --threads auto --project=. experiments/run_extremes.jl

# Generate figures
julia --threads auto --project=. plots/heatmaps.jl
```

Results are written to `results/` as CSVs. Figures are written to `plots/` as PDF and PNG.

## Dependencies

```toml
Graphs, CSV, DataFrames, CairoMakie
LinearAlgebra, Random, Statistics  # stdlib
```

## Reproducibility

All stochastic functions take an explicit `rng::AbstractRNG` argument. Experiment scripts seed with `MersenneTwister(42)` and derive per-replication seeds deterministically. Re-running any script produces identical results.
