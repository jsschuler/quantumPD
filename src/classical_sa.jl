using Random

function run_sa(Q, offset, n, rng;
                T0=2.0, alpha=0.995, max_steps=10_000, T_min=0.01)
    config = rand(rng, 0:1, n)
    current_welfare = welfare_from_config(config, Q, offset)
    best_config = copy(config)
    best_welfare = current_welfare
    trajectory = Float64[]

    temp = T0
    for step in 1:max_steps
        temp < T_min && break
        # Flip a random bit
        idx = rand(rng, 1:n)
        config[idx] = 1 - config[idx]
        new_welfare = welfare_from_config(config, Q, offset)
        delta = new_welfare - current_welfare
        if delta >= 0 || rand(rng) < exp(delta / temp)
            current_welfare = new_welfare
            push!(trajectory, current_welfare)
            if current_welfare > best_welfare
                best_welfare = current_welfare
                best_config = copy(config)
            end
        else
            config[idx] = 1 - config[idx]  # revert
        end
        temp *= alpha
    end
    return best_config, best_welfare, trajectory
end

function run_sa_ensemble(Q, offset, n, n_runs, base_rng)
    configs = zeros(Int, n, n_runs)
    welfares = zeros(Float64, n_runs)
    for r in 1:n_runs
        rng = Random.MersenneTwister(rand(base_rng, UInt32))
        cfg, w, _ = run_sa(Q, offset, n, rng)
        configs[:, r] = cfg
        welfares[r] = w
    end
    return configs, welfares
end
