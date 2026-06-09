using Graphs

function build_qubo(g::SimpleGraph, R::Int, S::Int, T::Int, P::Int)
    n = nv(g)
    Q = zeros(Float64, n, n)
    # Diagonal: (R - S) * deg(i) for each node
    for i in 1:n
        Q[i, i] = (R - S) * degree(g, i)
    end
    # Off-diagonal: -(R - 2S + P) for each edge (upper triangle)
    coeff = -(R - 2*S + P)
    for e in edges(g)
        i, j = src(e), dst(e)
        if i < j
            Q[i, j] = coeff
        else
            Q[j, i] = coeff
        end
    end
    offset = -R * ne(g)
    return Q, Float64(offset)
end

function welfare_from_config(config::Vector{Int}, Q::Matrix{Float64}, offset::Float64)::Float64
    n = length(config)
    val = 0.0
    for i in 1:n
        val += Q[i, i] * config[i]
        for j in (i+1):n
            val += Q[i, j] * config[i] * config[j]
        end
    end
    return -(val + offset)
end
