using Graphs
using LinearAlgebra
using Random

function generate_erdos_renyi(n::Int, p::Float64, rng::AbstractRNG)::SimpleGraph
    g = SimpleGraph(n)
    for i in 1:n, j in (i+1):n
        if rand(rng) < p
            add_edge!(g, i, j)
        end
    end
    return g
end

function generate_watts_strogatz(n::Int, k::Int, beta::Float64, rng::AbstractRNG)::SimpleGraph
    @assert iseven(k) "k must be even"
    g = SimpleGraph(n)
    half = k ÷ 2
    # Build ring lattice
    for i in 1:n, r in 1:half
        j = mod1(i + r, n)
        add_edge!(g, i, j)
    end
    # Rewire
    for i in 1:n, r in 1:half
        j = mod1(i + r, n)
        if rand(rng) < beta
            rem_edge!(g, i, j)
            # Pick new target not already connected and not self
            candidates = setdiff(1:n, neighbors(g, i), [i])
            if !isempty(candidates)
                new_j = rand(rng, candidates)
                add_edge!(g, i, new_j)
            else
                # Re-add original edge if no candidate
                add_edge!(g, i, j)
            end
        end
    end
    return g
end

function spectral_gap(g::SimpleGraph)::Float64
    n = nv(g)
    n == 0 && return 0.0
    if !is_connected(g)
        return 0.0
    end
    deg = degree(g)
    # Normalized Laplacian: L_norm[i,j] = -1/sqrt(d_i*d_j) for edge (i,j), 1 for i==j
    L = zeros(Float64, n, n)
    for i in 1:n
        L[i, i] = 1.0
    end
    for e in edges(g)
        i, j = src(e), dst(e)
        val = -1.0 / sqrt(deg[i] * deg[j])
        L[i, j] = val
        L[j, i] = val
    end
    evals = eigvals(Symmetric(L))
    sort!(evals)
    return evals[2]
end

function clustering_coefficient(g::SimpleGraph)::Float64
    n = nv(g)
    triangles = 0
    triplets = 0
    for v in 1:n
        nbrs = neighbors(g, v)
        d = length(nbrs)
        triplets += d * (d - 1)
        for i in 1:length(nbrs), j in (i+1):length(nbrs)
            if has_edge(g, nbrs[i], nbrs[j])
                triangles += 2
            end
        end
    end
    triplets == 0 && return 0.0
    return triangles / triplets
end

function frustration_index(g::SimpleGraph)::Float64
    # Count triangles (i,j,k) with all edges present
    # A triangle is "frustrated" if no assignment of {C,D} makes all three
    # nodes prefer cooperation simultaneously. In a symmetric welfare PD,
    # all triangles are frustrated (due to the defection incentive), so
    # frustration = number of frustrated triangles / total triangles.
    # Here we use the fraction of triangles, since each triangle is frustrated.
    n = nv(g)
    total_triangles = 0
    for i in 1:n
        nbrs_i = Set(neighbors(g, i))
        for j in neighbors(g, i)
            j <= i && continue
            nbrs_j = Set(neighbors(g, j))
            common = intersect(nbrs_i, nbrs_j)
            for k in common
                k <= j && continue
                total_triangles += 1
            end
        end
    end
    total_triangles == 0 && return 0.0
    # All triangles are frustrated in the PD welfare problem
    return 1.0
end
