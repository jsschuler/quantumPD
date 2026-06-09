using Statistics

function logical_entropy(config::Vector{Int}, n::Int)::Float64
    n_defect = sum(config)
    n_coop = n - n_defect
    pc = n_coop / n
    pd = n_defect / n
    return 2.0 * pc * pd
end

function approximation_ratio(achieved::Float64, optimal::Float64)::Float64
    optimal == 0.0 && return achieved >= 0.0 ? 1.0 : 0.0
    return achieved / optimal
end

function solution_coincidence(config::Vector{Int}, optimal_config::Vector{Int})::Bool
    return config == optimal_config
end

function cooperation_rate(config::Vector{Int})::Float64
    return (length(config) - sum(config)) / length(config)
end

function summarize_ensemble(configs, welfares, optimal_welfare, optimal_config, n)
    n_runs = length(welfares)
    mean_w = mean(welfares)
    std_w = std(welfares)
    approx_ratios = [approximation_ratio(welfares[r], optimal_welfare) for r in 1:n_runs]
    mean_ar = mean(approx_ratios)
    coincidences = [solution_coincidence(configs[:, r], optimal_config) for r in 1:n_runs]
    coin_rate = mean(coincidences)
    coop_rates = [cooperation_rate(configs[:, r]) for r in 1:n_runs]
    mean_coop = mean(coop_rates)
    entropies = [logical_entropy(configs[:, r], n) for r in 1:n_runs]
    mean_ent = mean(entropies)
    return (
        mean_welfare = mean_w,
        std_welfare = std_w,
        mean_approx_ratio = mean_ar,
        coincidence_rate = coin_rate,
        mean_cooperation = mean_coop,
        mean_logical_entropy = mean_ent,
    )
end
