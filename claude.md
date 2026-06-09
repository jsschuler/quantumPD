# CLAUDE.md — Network Prisoner's Dilemma: Classical vs Quantum Annealing

## Project overview

This project implements a computational experiment comparing classical simulated
annealing (SA), simulated quantum annealing (SQA), and exact brute-force solution
on a network prisoner's dilemma (PD) welfare maximization problem. The experiment
varies network topology across Erdős-Rényi and Watts-Strogatz random graph models,
plus two structural extremes (complete graph, balanced bipartite graph). The primary
research question is whether and where the welfare optimum found by quantum annealing
diverges from classical SA, and how this divergence depends on graph structure.

## Language and environment

- Julia 1.10+
- All dependencies managed via `Project.toml` and `Manifest.toml`
- No Python, no Jupyter. Plain `.jl` scripts only.
- Entry point for each experiment is a standalone script in `experiments/`

## Directory structure

```
quantum_pd/
├── CLAUDE.md
├── Project.toml
├── src/
│   ├── QuantumPD.jl        # top-level module, exports all public functions
│   ├── graphs.jl           # graph generation and structural metrics
│   ├── qubo.jl             # QUBO encoding of network PD welfare
│   ├── exact_solver.jl     # brute force over all 2^n configurations
│   ├── classical_sa.jl     # simulated annealing
│   ├── sqa.jl              # simulated quantum annealing (SQA)
│   ├── exact_quantum.jl    # exact Schrödinger evolution via matrix exponentiation
│   └── metrics.jl          # solution quality metrics and logical entropy
├── experiments/
│   ├── run_erdos_renyi.jl  # sweep over connection probability p
│   ├── run_watts_strogatz.jl  # sweep over rewiring probability beta
│   └── run_extremes.jl     # complete graph K_10 and bipartite K_{5,5}
├── results/                # CSV outputs, one file per experiment
└── plots/
    └── heatmaps.jl         # generate all figures from results CSVs
```

## Fixed experimental parameters

These are fixed for round 1 and must not be varied without explicit instruction:

```julia
const N_NODES       = 10
const R             = 3   # reward (mutual cooperation)
const S             = 0   # sucker payoff
const T             = 5   # temptation (unilateral defection)
const P             = 1   # punishment (mutual defection)
const N_REPLICATIONS = 50  # random graph instances per parameter point
const N_SA_RUNS     = 100  # SA runs per instance
const N_SQA_RUNS    = 100  # SQA runs per instance
```

Payoffs satisfy the PD conditions: T > R > P > S and 2R > T + S.

### Erdős-Rényi sweep

```julia
const ER_P_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]
```

### Watts-Strogatz sweep

```julia
const WS_K    = 4                          # each node initially connected to k nearest neighbors
const WS_BETA = [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Extremes

- Complete graph: `complete_graph(10)` from Graphs.jl
- Balanced bipartite: `complete_bipartite_graph(5, 5)` from Graphs.jl
- Both are deterministic — no replications needed, run N_SA_RUNS and N_SQA_RUNS directly

## Dependencies

```toml
[deps]
Graphs = "*"
LinearAlgebra = "*"   # stdlib
Random = "*"          # stdlib
Statistics = "*"      # stdlib
CSV = "*"
DataFrames = "*"
CairoMakie = "*"
```

Do not add dependencies beyond this list without a clear reason. In particular,
do not add any quantum computing SDK — quantum simulation is implemented from
scratch using LinearAlgebra.

## Module: graphs.jl

### Functions to implement

```julia
"""
    generate_erdos_renyi(n, p, rng) -> SimpleGraph

Generate an Erdős-Rényi random graph G(n,p).
Use the provided rng for reproducibility.
"""
function generate_erdos_renyi(n::Int, p::Float64, rng::AbstractRNG)::SimpleGraph

"""
    generate_watts_strogatz(n, k, beta, rng) -> SimpleGraph

Generate a Watts-Strogatz small-world graph.
Start from a k-regular ring lattice, rewire each edge with probability beta.
k must be even.
"""
function generate_watts_strogatz(n::Int, k::Int, beta::Float64, rng::AbstractRNG)::SimpleGraph

"""
    spectral_gap(g) -> Float64

Return the algebraic connectivity (second-smallest eigenvalue of the
normalized graph Laplacian). Zero for disconnected graphs.
"""
function spectral_gap(g::SimpleGraph)::Float64

"""
    clustering_coefficient(g) -> Float64

Global clustering coefficient: ratio of closed triplets to all triplets.
"""
function clustering_coefficient(g::SimpleGraph)::Float64

"""
    frustration_index(g) -> Float64

Fraction of triangles in g that are frustrated under the all-cooperate
assignment. For the PD welfare problem, a triangle (i,j,k) is frustrated
if no assignment of {C,D} to all three nodes simultaneously satisfies
all three pairwise cooperation incentives.
In the symmetric welfare PD this equals the fraction of odd cycles
normalized by total triangles. For a bipartite graph this is always 0.
"""
function frustration_index(g::SimpleGraph)::Float64
```

## Module: qubo.jl

### QUBO encoding

The network PD welfare maximization problem: given graph $G = (V, E)$ and binary
strategies $s_i \in \{0,1\}$ (0 = Cooperate, 1 = Defect), maximize total welfare:

$$W(\mathbf{s}) = \sum_{(i,j) \in E} w_{ij}(s_i, s_j)$$

where the edge welfare contribution is:

$$w_{ij}(s_i, s_j) = R(1-s_i)(1-s_j) + S[(1-s_i)s_j + s_i(1-s_j)] + P s_i s_j$$

Note S appears symmetrically because welfare sums both players' payoffs and the
asymmetric T payoff cancels: $\pi_i(C,D) + \pi_j(C,D) = S + T = S + T$. Verify
this cancellation explicitly in the implementation.

Converting to minimization (QUBO convention), $\min -W(\mathbf{s})$:

The QUBO matrix $Q$ is $n \times n$ with:

$$Q_{ii} = (R - S) \cdot \deg(i) \quad \text{(diagonal)}$$
$$Q_{ij} = -(R - 2S + P) \quad \text{for each edge } (i,j) \text{ (upper triangle)}$$

The constant offset $-R \cdot |E|$ is tracked separately for correct welfare recovery.

```julia
"""
    build_qubo(g, R, S, T, P) -> (Q::Matrix{Float64}, offset::Float64)

Return the QUBO matrix Q and constant offset such that
    -W(s) = s' * Q * s + offset
for binary vector s ∈ {0,1}^n.
Upper triangular storage: Q[i,j] for i < j holds the edge coefficient.
Diagonal Q[i,i] holds the linear term for node i.
"""
function build_qubo(g::SimpleGraph, R::Int, S::Int, T::Int, P::Int)

"""
    welfare_from_config(config, Q, offset) -> Float64

Given a binary configuration vector config ∈ {0,1}^n,
return the welfare W = -(config' * Q * config + offset).
"""
function welfare_from_config(config::Vector{Int}, Q::Matrix{Float64}, offset::Float64)::Float64
```

## Module: exact_solver.jl

Brute force over all $2^n$ configurations. Feasible for $n = 10$ (1024 configs).

```julia
"""
    exact_solve(Q, offset, n) -> (best_config::Vector{Int}, best_welfare::Float64)

Enumerate all 2^n binary configurations, return the one maximizing welfare.
In case of ties, return the configuration with the most cooperation (fewest 1s).
"""
function exact_solve(Q::Matrix{Float64}, offset::Float64, n::Int)
```

## Module: classical_sa.jl

Standard simulated annealing on binary configurations.

### Cooling schedule

Use geometric cooling: $T_k = T_0 \cdot \alpha^k$ where:
- $T_0 = 2.0$ (initial temperature — set to be on the order of the largest welfare
  difference between neighboring configurations)
- $\alpha = 0.995$ (cooling rate)
- Terminate when $T_k < 0.01$ or after 10,000 steps, whichever comes first

### Proposal distribution

At each step, flip a single randomly chosen bit (single spin flip).

```julia
"""
    run_sa(Q, offset, n, rng;
           T0=2.0, alpha=0.995, max_steps=10_000, T_min=0.01)
    -> (best_config::Vector{Int}, best_welfare::Float64, welfare_trajectory::Vector{Float64})

Run one SA trajectory. Return best configuration found, its welfare,
and the welfare at each accepted step (for convergence diagnostics).
"""
function run_sa(Q, offset, n, rng; T0=2.0, alpha=0.995, max_steps=10_000, T_min=0.01)

"""
    run_sa_ensemble(Q, offset, n, n_runs, base_rng)
    -> (configs::Matrix{Int}, welfares::Vector{Float64})

Run n_runs independent SA trajectories with different RNG seeds derived from base_rng.
Return all final configurations (n x n_runs matrix) and final welfares.
"""
function run_sa_ensemble(Q, offset, n, n_runs, base_rng)
```

## Module: sqa.jl

Simulated Quantum Annealing via path-integral Monte Carlo (Trotter decomposition).

### Method

The quantum system is simulated by $P$ Trotter slices (replicas) of the classical
system, coupled along the imaginary time dimension. The effective Hamiltonian is:

$$H_{\text{eff}} = \frac{1}{P} \sum_{\tau=1}^{P} H_{\text{classical}}(\mathbf{s}^\tau)
- J_\perp \sum_{\tau=1}^{P} \sum_{i=1}^{n} s_i^\tau s_i^{\tau+1}$$

where $J_\perp = \frac{P}{2\beta} \ln \coth(\beta \Gamma / P)$ is the inter-replica
coupling encoding the transverse field $\Gamma$.

The transverse field $\Gamma$ is annealed from $\Gamma_0$ to $0$ over the annealing
schedule, while inverse temperature $\beta$ is held fixed.

### Parameters

```julia
const SQA_P        = 20     # number of Trotter slices
const SQA_BETA     = 10.0   # inverse temperature (fixed during annealing)
const SQA_GAMMA_0  = 3.0    # initial transverse field
const SQA_STEPS    = 10_000 # Monte Carlo steps per run
```

### Update rule

At each MC step, propose a single spin flip at a random site in a random Trotter
slice. Accept with Metropolis probability accounting for both the classical energy
change within the slice and the inter-replica coupling change along the Trotter
dimension.

```julia
"""
    run_sqa(Q, offset, n, rng) -> (best_config::Vector{Int}, best_welfare::Float64)

Run one SQA trajectory. The best configuration is the Trotter slice with
highest welfare at the end of annealing.
"""
function run_sqa(Q, offset, n, rng)

"""
    run_sqa_ensemble(Q, offset, n, n_runs, base_rng)
    -> (configs::Matrix{Int}, welfares::Vector{Float64})

Run n_runs independent SQA trajectories.
"""
function run_sqa_ensemble(Q, offset, n, n_runs, base_rng)
```

## Module: exact_quantum.jl

Exact simulation of quantum annealing via direct Schrödinger evolution.
Feasible for $n = 10$ ($2^{10} = 1024$ dimensional Hilbert space).

### Hamiltonian

The time-dependent Hamiltonian interpolates between transverse field and problem:

$$H(s) = -(1-s) \Gamma \sum_i \sigma_i^x - s \sum_{i < j} J_{ij} \sigma_i^z \sigma_j^z$$

where $s = t/T \in [0,1]$ is the normalized annealing time, and $J_{ij}$ are the
QUBO couplings (note sign: we are minimizing energy, so problem Hamiltonian uses
the negated QUBO matrix entries as couplings).

### Implementation

Build the full $2^n \times 2^n$ Hamiltonian matrices using Kronecker products of
Pauli matrices. Use the adiabatic approximation: at each of $M$ discrete time steps,
compute the instantaneous ground state by exact diagonalization and track the
overlap of the evolving state with the instantaneous ground state.

For the final state, measure in the computational basis by sampling from
$|\langle \mathbf{s} | \psi(T) \rangle|^2$.

```julia
"""
    pauli_x(n, i) -> Matrix{ComplexF64}

Return the n-qubit operator σ_i^x = I ⊗ ... ⊗ σ^x ⊗ ... ⊗ I
with σ^x at position i (1-indexed).
"""
function pauli_x(n::Int, i::Int)::Matrix{ComplexF64}

"""
    pauli_z(n, i) -> Matrix{ComplexF64}

Return the n-qubit operator σ_i^z.
"""
function pauli_z(n::Int, i::Int)::Matrix{ComplexF64}

"""
    build_hamiltonian(s, Q, n, Gamma) -> Matrix{ComplexF64}

Build the full 2^n × 2^n Hamiltonian at annealing parameter s ∈ [0,1].
"""
function build_hamiltonian(s::Float64, Q::Matrix{Float64}, n::Int, Gamma::Float64)

"""
    run_exact_quantum(Q, offset, n;
                      Gamma=3.0, n_steps=100, n_samples=1000, rng=Random.default_rng())
    -> (best_config::Vector{Int}, best_welfare::Float64,
        ground_state_overlap::Vector{Float64})

Simulate adiabatic quantum annealing exactly.
- Evolve the state through n_steps Hamiltonian evaluations
- At each step use the exact ground state (adiabatic limit)
- Track ground_state_overlap: overlap of instantaneous state with ground state
- At the end, sample n_samples measurement outcomes from the final state distribution
- Return the highest-welfare sampled configuration
"""
function run_exact_quantum(Q, offset, n; Gamma=3.0, n_steps=100, n_samples=1000, rng=Random.default_rng())
```

## Module: metrics.jl

```julia
"""
    logical_entropy(config, n) -> Float64

Compute the logical entropy h(π) of the coordination partition induced by config.
The partition has two blocks: cooperators {i : s_i = 0} and defectors {i : s_i = 1}.
h(π) = 1 - Pr(C)^2 - Pr(D)^2 = 2 * Pr(C) * Pr(D)
where Pr(C) = (number of cooperators) / n.

Note: this is the two-block partition entropy. It equals zero when all agents
are in the same block (all-C or all-D) and is maximized at 0.5 when exactly
half cooperate.
"""
function logical_entropy(config::Vector{Int}, n::Int)::Float64

"""
    approximation_ratio(achieved_welfare, optimal_welfare) -> Float64

Return achieved_welfare / optimal_welfare. Should be in (0, 1].
"""
function approximation_ratio(achieved::Float64, optimal::Float64)::Float64

"""
    solution_coincidence(config, optimal_config) -> Bool

Return true if config == optimal_config (exact match with welfare optimum).
"""
function solution_coincidence(config::Vector{Int}, optimal_config::Vector{Int})::Bool

"""
    cooperation_rate(config) -> Float64

Fraction of agents cooperating (s_i = 0).
"""
function cooperation_rate(config::Vector{Int})::Float64

"""
    summarize_ensemble(configs, welfares, optimal_welfare, optimal_config, n)
    -> NamedTuple

Given an ensemble of runs, return:
  - mean_welfare: mean welfare across runs
  - std_welfare: std of welfare across runs
  - mean_approx_ratio: mean approximation ratio
  - coincidence_rate: fraction of runs that found the exact optimum
  - mean_cooperation: mean cooperation rate across runs
  - mean_logical_entropy: mean logical entropy across runs
"""
function summarize_ensemble(configs, welfares, optimal_welfare, optimal_config, n)
```

## Experiment scripts

### experiments/run_erdos_renyi.jl

For each $p \in$ `ER_P_VALUES`:
1. Generate `N_REPLICATIONS` random graphs using sequential RNG seeds
2. For each graph:
   a. Build QUBO
   b. Exact solve → optimal welfare and config
   c. Run SA ensemble → summary statistics
   d. Run SQA ensemble → summary statistics
   e. Run exact quantum → best config and ground state overlap
   f. Compute graph metrics: spectral gap, clustering coefficient, frustration index,
      number of edges
3. Collect all results into a DataFrame
4. Write to `results/erdos_renyi.csv`

Schema for `erdos_renyi.csv`:

```
p, replication, n_edges, spectral_gap, clustering, frustration,
optimal_welfare,
sa_mean_welfare, sa_std_welfare, sa_approx_ratio, sa_coincidence_rate,
sa_mean_cooperation, sa_mean_logical_entropy,
sqa_mean_welfare, sqa_std_welfare, sqa_approx_ratio, sqa_coincidence_rate,
sqa_mean_cooperation, sqa_mean_logical_entropy,
eq_welfare, eq_approx_ratio, eq_coincidence, eq_ground_state_overlap,
eq_cooperation, eq_logical_entropy
```

### experiments/run_watts_strogatz.jl

Identical structure to `run_erdos_renyi.jl` but sweeps over `WS_BETA` values.
Output to `results/watts_strogatz.csv` with same schema plus column `beta`
replacing `p`.

### experiments/run_extremes.jl

Run complete graph $K_{10}$ and balanced bipartite $K_{5,5}$ without replications.
Run `N_SA_RUNS` SA trajectories and `N_SQA_RUNS` SQA trajectories directly.
Run exact quantum once.
Output to `results/extremes.csv`.

Also, for each extreme graph, save the full welfare trajectory of a single
representative SA run (for convergence plot).

## plots/heatmaps.jl

Generate the following figures from the results CSVs. Use CairoMakie.

**Figure 1: Erdős-Rényi approximation ratio heatmap**
X-axis: p values. Y-axis: metric (SA approx ratio, SQA approx ratio, exact quantum
approx ratio). Three-panel figure. Each bar is mean ± std across replications.

**Figure 2: Erdős-Rényi coincidence rate**
Same structure, showing coincidence rate (fraction of runs finding exact optimum).

**Figure 3: Watts-Strogatz approximation ratio**
Same as Figure 1 but for WS beta sweep.

**Figure 4: Welfare gap**
For both ER and WS: plot $W^* - \bar{W}_{SA}$ and $W^* - \bar{W}_{SQA}$ as
functions of p and beta. This is the key figure showing where quantum annealing
outperforms classical SA.

**Figure 5: Spectral gap vs welfare gap scatter**
X-axis: spectral gap. Y-axis: welfare gap (SA and SQA separately). One point per
graph instance. Should show that low spectral gap (frustrated networks) correlates
with large welfare gap and larger quantum advantage.

**Figure 6: Logical entropy comparison**
Mean logical entropy of SA, SQA, and exact quantum solutions vs p and beta.
Lower logical entropy = more coordination (more agents in same block).

**Figure 7: Extremes**
Bar chart comparing SA, SQA, and exact quantum on complete and bipartite graphs.
Show welfare, coincidence, cooperation rate, and logical entropy side by side.

All figures saved to `plots/` as PDF and PNG at 300 DPI.

## Implementation notes

### RNG discipline
Every function that uses randomness takes an explicit `rng::AbstractRNG` argument.
Experiment scripts seed with `Random.MersenneTwister(42)` for the base seed and
derive per-replication seeds as `Random.MersenneTwister(42 + replication_index)`.
This ensures full reproducibility.

### Type stability
All core functions should be type-stable. Use `@code_warntype` to verify before
considering the module complete.

### Testing
Before running experiments, verify:
1. `welfare_from_config` on all-C config gives $R \times |E|$ for any graph
2. `welfare_from_config` on all-D config gives $P \times |E|$ for any graph
3. `exact_solve` on complete bipartite graph returns all-C config
4. `logical_entropy` returns 0.0 for all-C and all-D configs
5. `logical_entropy` returns 0.5 for exactly half-cooperating config with n=10
6. SA and SQA on a single-edge two-node PD return cooperation with high probability
7. Exact quantum on a single-edge two-node PD returns cooperation with probability 1

Write these as a `test/runtests.jl` file using Julia's built-in `Test` module.

### Performance expectations
With n=10, N_REPLICATIONS=50, N_SA_RUNS=100, N_SQA_RUNS=100:
- Exact solver: negligible
- SA ensemble: < 1 second per graph instance
- SQA ensemble: < 10 seconds per graph instance
- Exact quantum: < 5 seconds per graph instance
- Full ER sweep: < 30 minutes
- Full WS sweep: < 30 minutes

If SQA is substantially slower than this, reduce SQA_STEPS before reducing
N_SQA_RUNS — ensemble size matters more than trajectory length for statistics.

## What success looks like

The experiment is successful if it produces clean versions of Figures 4 and 5 showing:

1. **Bipartite graph:** SA coincidence rate ≈ 1.0, SQA coincidence rate ≈ 1.0,
   welfare gap ≈ 0 for both. Both methods find the optimum reliably.

2. **Complete graph:** SA coincidence rate high (conjecture), SQA coincidence rate
   high. Welfare gap small for both.

3. **Intermediate ER density (p ≈ 0.4-0.6):** SA coincidence rate drops below 1.0,
   SQA coincidence rate remains higher. Welfare gap opens between SA and SQA.
   This is the main empirical finding.

4. **Spectral gap correlation:** Figure 5 shows negative correlation between spectral
   gap and welfare gap, confirming that frustrated (low spectral gap) networks are
   harder for SA and where quantum advantage is largest.

If finding 3 is not observed, the experiment is still informative — it would suggest
that for n=10 PD networks, classical SA is sufficient and the quantum advantage only
appears at larger n. This is also a publishable finding.