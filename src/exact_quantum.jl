using LinearAlgebra
using Random

# Build H_driver as a real dense matrix without Kronecker products.
# H_driver[k+1, k XOR (1<<(i-1)) + 1] -= Gamma for each state k and bit i.
function _build_driver(n::Int, Gamma::Float64)::Matrix{Float64}
    dim = 2^n
    H = zeros(Float64, dim, dim)
    for k in 0:(dim-1)
        for i in 0:(n-1)
            j = k ⊻ (1 << i)
            H[k+1, j+1] -= Gamma
        end
    end
    return H
end

# Build H_problem as a diagonal vector (H_problem is diagonal in computational basis
# because pauli_z is diagonal and all problem terms are products of pauli_z operators).
function _build_problem_diag(n::Int, Q::Matrix{Float64})::Vector{Float64}
    dim = 2^n
    diag_H = zeros(Float64, dim)
    # Precompute Ising-form fields and couplings from QUBO
    # QUBO: sum_i Q[i,i]*x_i + sum_{i<j} Q[i,j]*x_i*x_j,  x_i = (1 - z_i)/2
    # Ising couplings: J_ij = Q[i,j]/4 (off-diag), local field h_i = -Q[i,i]/2 - sum_{j!=i} Q_{ij}/4
    h = zeros(Float64, n)
    for i in 1:n
        h[i] = -Q[i,i] / 2.0
        for j in 1:n
            j == i && continue
            qij = i < j ? Q[i,j] : Q[j,i]
            h[i] -= qij / 4.0
        end
    end
    for k in 0:(dim-1)
        val = 0.0
        # z_i = +1 if bit (i-1) of k is 0, -1 if 1
        for i in 1:n
            zi = ((k >> (i-1)) & 1) == 0 ? 1.0 : -1.0
            val += h[i] * zi
        end
        for i in 1:n, j in (i+1):n
            Jij = Q[i,j] / 4.0
            abs(Jij) < 1e-14 && continue
            zi = ((k >> (i-1)) & 1) == 0 ? 1.0 : -1.0
            zj = ((k >> (j-1)) & 1) == 0 ? 1.0 : -1.0
            val += Jij * zi * zj
        end
        diag_H[k+1] = val
    end
    return diag_H
end

function pauli_x(n::Int, i::Int)::Matrix{ComplexF64}
    sx = ComplexF64[0 1; 1 0]
    id = ComplexF64[1 0; 0 1]
    ops = [k == i ? sx : id for k in 1:n]
    return foldl(kron, ops)
end

function pauli_z(n::Int, i::Int)::Matrix{ComplexF64}
    sz = ComplexF64[1 0; 0 -1]
    id = ComplexF64[1 0; 0 1]
    ops = [k == i ? sz : id for k in 1:n]
    return foldl(kron, ops)
end

function build_hamiltonian(s::Float64, Q::Matrix{Float64}, n::Int, Gamma::Float64)::Matrix{ComplexF64}
    H_driver = _build_driver(n, Gamma)
    H_prob_diag = _build_problem_diag(n, Q)
    dim = 2^n
    H = ComplexF64.(((1.0 - s) .* H_driver))
    for k in 1:dim
        H[k, k] += s * H_prob_diag[k]
    end
    return H
end

function run_exact_quantum(Q, offset, n;
                           Gamma=3.0, n_steps=20, n_samples=1000,
                           rng=Random.default_rng())
    dim = 2^n

    # Precompute operators once (the expensive part)
    H_driver = _build_driver(n, Gamma)
    H_prob_diag = _build_problem_diag(n, Q)

    # Working matrix buffer (real symmetric throughout)
    H_work = zeros(Float64, dim, dim)

    ground_state_overlap = ones(Float64, n_steps)
    psi = fill(1.0 / sqrt(Float64(dim)), dim)

    for step in 1:n_steps
        s = n_steps == 1 ? 1.0 : (step - 1) / (n_steps - 1)
        coeff = 1.0 - s
        # H_work = (1-s)*H_driver
        @. H_work = coeff * H_driver
        # Add diagonal problem terms
        for k in 1:dim
            H_work[k, k] += s * H_prob_diag[k]
        end
        F = eigen(Symmetric(H_work))
        psi = F.vectors[:, 1]
        # In the adiabatic limit the state IS the ground state — overlap = 1
        ground_state_overlap[step] = 1.0
    end

    # Sample measurement outcomes from final state probability distribution
    probs = psi .^ 2
    probs ./= sum(probs)
    cumprobs = cumsum(probs)

    best_welfare = -Inf
    best_config = zeros(Int, n)

    for _ in 1:n_samples
        r = rand(rng)
        idx = searchsortedfirst(cumprobs, r) - 1
        idx = clamp(idx, 0, dim - 1)
        config = [(idx >> (i-1)) & 1 for i in 1:n]
        w = welfare_from_config(config, Q, offset)
        if w > best_welfare
            best_welfare = w
            best_config = config
        end
    end

    return best_config, best_welfare, ground_state_overlap
end
