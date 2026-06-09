using Random

const SQA_P       = 20
const SQA_BETA    = 10.0
const SQA_GAMMA_0 = 3.0
const SQA_STEPS   = 10_000

function _jperp(beta::Float64, gamma::Float64, P::Int)::Float64
    arg = beta * gamma / P
    # J_perp = (P / (2*beta)) * ln(coth(arg))
    arg <= 0.0 && return 0.0
    coth_val = cosh(arg) / sinh(arg)
    return (P / (2.0 * beta)) * log(coth_val)
end

function run_sqa(Q, offset, n, rng)
    P = SQA_P
    beta = SQA_BETA
    gamma0 = SQA_GAMMA_0
    steps = SQA_STEPS

    # Initialize Trotter replicas randomly
    replicas = [rand(rng, 0:1, n) for _ in 1:P]

    best_welfare = -Inf
    best_config = copy(replicas[1])

    for step in 1:steps
        s = step / steps
        gamma = gamma0 * (1.0 - s)
        jperp = _jperp(beta, gamma, P)

        # Pick random Trotter slice and random spin
        tau = rand(rng, 1:P)
        i = rand(rng, 1:n)

        # Energy change in classical part (within slice)
        # delta_classical = welfare change within Trotter slice tau
        config = replicas[tau]
        # Compute delta_E for flipping spin i in slice tau (note: we minimize -W/P)
        old_val = config[i]
        new_val = 1 - old_val

        # Classical energy contribution from slice: (1/P) * (-W)
        # delta = (1/P) * (new_energy - old_energy) where energy = -W (minimizing)
        # so delta = -(1/P) * (new_welfare - old_welfare)
        w_old = welfare_from_config(config, Q, offset)
        config[i] = new_val
        w_new = welfare_from_config(config, Q, offset)
        config[i] = old_val

        delta_classical = (1.0 / P) * (-(w_new - w_old))  # cost change (minimizing)

        # Inter-replica coupling energy change
        tau_prev = mod1(tau - 1, P)
        tau_next = mod1(tau + 1, P)
        s_prev = replicas[tau_prev][i]
        s_next = replicas[tau_next][i]
        # coupling term: -J_perp * s_i^tau * s_i^(tau+1) summed over tau
        # change when s_i^tau: old_val -> new_val
        old_coupling = -jperp * (old_val * s_prev + old_val * s_next)
        new_coupling = -jperp * (new_val * s_prev + new_val * s_next)
        delta_coupling = new_coupling - old_coupling

        delta_total = delta_classical + delta_coupling

        if delta_total <= 0.0 || rand(rng) < exp(-beta * delta_total)
            replicas[tau][i] = new_val
        end
    end

    # Find best Trotter slice
    for tau in 1:P
        w = welfare_from_config(replicas[tau], Q, offset)
        if w > best_welfare
            best_welfare = w
            best_config = copy(replicas[tau])
        end
    end

    return best_config, best_welfare
end

function run_sqa_ensemble(Q, offset, n, n_runs, base_rng)
    configs = zeros(Int, n, n_runs)
    welfares = zeros(Float64, n_runs)
    for r in 1:n_runs
        rng = Random.MersenneTwister(rand(base_rng, UInt32))
        cfg, w = run_sqa(Q, offset, n, rng)
        configs[:, r] = cfg
        welfares[r] = w
    end
    return configs, welfares
end
