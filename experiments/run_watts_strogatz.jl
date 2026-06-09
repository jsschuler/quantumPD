push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using QuantumPD
using Random
using DataFrames
using CSV
using Statistics
using Base.Threads

const N_NODES        = 10
const R              = 3
const S              = 0
const T              = 5
const P_PAY          = 1
const N_REPLICATIONS = 50
const N_SA_RUNS      = 100
const N_SQA_RUNS     = 100
const WS_K           = 4
const WS_BETA        = [0.0, 0.25, 0.5, 0.75, 1.0]

println("Threads: ", nthreads())

rows = []
lock = ReentrantLock()

for beta in WS_BETA
    println("WS beta=$beta")
    local_rows = Vector{Any}(undef, N_REPLICATIONS)

    @threads for rep in 1:N_REPLICATIONS
        rng = Random.MersenneTwister(42 + rep)

        g = generate_watts_strogatz(N_NODES, WS_K, beta, rng)
        Q, offset = build_qubo(g, R, S, T, P_PAY)
        n = nv(g)

        opt_config, opt_welfare = exact_solve(Q, offset, n)

        sa_rng = Random.MersenneTwister(1000 + rep)
        sa_configs, sa_welfares = run_sa_ensemble(Q, offset, n, N_SA_RUNS, sa_rng)
        sa_stats = summarize_ensemble(sa_configs, sa_welfares, opt_welfare, opt_config, n)

        sqa_rng = Random.MersenneTwister(2000 + rep)
        sqa_configs, sqa_welfares = run_sqa_ensemble(Q, offset, n, N_SQA_RUNS, sqa_rng)
        sqa_stats = summarize_ensemble(sqa_configs, sqa_welfares, opt_welfare, opt_config, n)

        eq_rng = Random.MersenneTwister(3000 + rep)
        eq_config, eq_welfare, eq_overlap = run_exact_quantum(Q, offset, n; rng=eq_rng)
        eq_ar = approximation_ratio(eq_welfare, opt_welfare)
        eq_coin = solution_coincidence(eq_config, opt_config)
        eq_coop = cooperation_rate(eq_config)
        eq_ent = logical_entropy(eq_config, n)
        eq_mean_overlap = mean(eq_overlap)

        sg = spectral_gap(g)
        cc = clustering_coefficient(g)
        fi = frustration_index(g)

        local_rows[rep] = (
            beta = beta,
            replication = rep,
            n_edges = ne(g),
            spectral_gap = sg,
            clustering = cc,
            frustration = fi,
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
            eq_ground_state_overlap = eq_mean_overlap,
            eq_cooperation = eq_coop,
            eq_logical_entropy = eq_ent,
        )
    end

    append!(rows, local_rows)
end

df = DataFrame(rows)
mkpath(joinpath(@__DIR__, "..", "results"))
CSV.write(joinpath(@__DIR__, "..", "results", "watts_strogatz.csv"), df)
println("Done. Wrote results/watts_strogatz.csv")
