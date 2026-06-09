push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using QuantumPD
using Random
using DataFrames
using CSV
using Statistics
using Base.Threads
using Graphs

const R       = 3
const S       = 0
const T       = 5
const P_PAY   = 1
const N_SA_RUNS  = 100
const N_SQA_RUNS = 100

println("Threads: ", nthreads())

graph_specs = [
    ("complete_K10",  complete_graph(10)),
    ("bipartite_K55", complete_bipartite_graph(5, 5)),
]

rows = Vector{Any}(undef, length(graph_specs))
trajectories = Dict{String, Vector{Float64}}()
traj_lock = ReentrantLock()

@threads for gidx in 1:length(graph_specs)
    name, g = graph_specs[gidx]
    println("Running $name")
    n = nv(g)
    Q, offset = build_qubo(g, R, S, T, P_PAY)

    opt_config, opt_welfare = exact_solve(Q, offset, n)

    sa_rng = Random.MersenneTwister(42)
    sa_configs, sa_welfares = run_sa_ensemble(Q, offset, n, N_SA_RUNS, sa_rng)
    sa_stats = summarize_ensemble(sa_configs, sa_welfares, opt_welfare, opt_config, n)

    rep_rng = Random.MersenneTwister(999)
    _, _, traj = run_sa(Q, offset, n, rep_rng)

    sqa_rng = Random.MersenneTwister(100)
    sqa_configs, sqa_welfares = run_sqa_ensemble(Q, offset, n, N_SQA_RUNS, sqa_rng)
    sqa_stats = summarize_ensemble(sqa_configs, sqa_welfares, opt_welfare, opt_config, n)

    eq_rng = Random.MersenneTwister(200)
    eq_config, eq_welfare, eq_overlap = run_exact_quantum(Q, offset, n; rng=eq_rng)
    eq_ar = approximation_ratio(eq_welfare, opt_welfare)
    eq_coin = solution_coincidence(eq_config, opt_config)
    eq_coop = cooperation_rate(eq_config)
    eq_ent = logical_entropy(eq_config, n)

    rows[gidx] = (
        graph = name,
        n_edges = ne(g),
        spectral_gap = spectral_gap(g),
        clustering = clustering_coefficient(g),
        frustration = frustration_index(g),
        optimal_welfare = opt_welfare,
        sa_mean_welfare = sa_stats.mean_welfare,
        sa_std_welfare = sa_stats.std_welfare,
        sa_approx_ratio = sa_stats.mean_approx_ratio,
        sa_coincidence_rate = sa_stats.coincidence_rate,
        sa_mean_cooperation = sa_stats.mean_cooperation,
        sa_mean_logical_entropy = sa_stats.mean_logical_entropy,
        sqa_mean_welfare = sqa_stats.mean_welfare,
        sqa_std_welfare = sqa_stats.std_welfare,
        sqa_approx_ratio = sqa_stats.mean_approx_ratio,
        sqa_coincidence_rate = sqa_stats.coincidence_rate,
        sqa_mean_cooperation = sqa_stats.mean_cooperation,
        sqa_mean_logical_entropy = sqa_stats.mean_logical_entropy,
        eq_welfare = eq_welfare,
        eq_approx_ratio = eq_ar,
        eq_coincidence = eq_coin,
        eq_ground_state_overlap = mean(eq_overlap),
        eq_cooperation = eq_coop,
        eq_logical_entropy = eq_ent,
    )

    lock(traj_lock) do
        trajectories[name] = traj
    end
end

mkpath(joinpath(@__DIR__, "..", "results"))
df = DataFrame(rows)
CSV.write(joinpath(@__DIR__, "..", "results", "extremes.csv"), df)

for (name, traj) in trajectories
    tdf = DataFrame(step=1:length(traj), welfare=traj)
    CSV.write(joinpath(@__DIR__, "..", "results", "trajectory_$(name).csv"), tdf)
end

println("Done. Wrote results/extremes.csv and trajectory files.")
