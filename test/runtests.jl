using Test
using Random
using Statistics
using Graphs

# Load module from source
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using QuantumPD

const R = 3; const S = 0; const T = 5; const P_pay = 1

@testset "welfare_from_config basic" begin
    # Complete graph K_4
    g = complete_graph(4)
    Q, offset = build_qubo(g, R, S, T, P_pay)
    n = nv(g)
    ne_g = ne(g)

    # All cooperate (all zeros)
    all_c = zeros(Int, n)
    @test welfare_from_config(all_c, Q, offset) ≈ R * ne_g

    # All defect (all ones)
    all_d = ones(Int, n)
    @test welfare_from_config(all_d, Q, offset) ≈ P_pay * ne_g
end

@testset "exact_solve bipartite" begin
    g = complete_bipartite_graph(5, 5)
    Q, offset = build_qubo(g, R, S, T, P_pay)
    n = nv(g)
    config, welfare = exact_solve(Q, offset, n)
    # All-cooperate should be optimal for bipartite (no odd cycles, no frustrated triangles)
    @test welfare ≈ R * ne(g)
    @test all(config .== 0)
end

@testset "logical_entropy" begin
    n = 10
    all_c = zeros(Int, n)
    all_d = ones(Int, n)
    half = vcat(zeros(Int, 5), ones(Int, 5))

    @test logical_entropy(all_c, n) ≈ 0.0
    @test logical_entropy(all_d, n) ≈ 0.0
    @test logical_entropy(half, n) ≈ 0.5
end

@testset "SA two-node PD" begin
    # Single edge, 2 nodes
    g = SimpleGraph(2)
    add_edge!(g, 1, 2)
    Q, offset = build_qubo(g, R, S, T, P_pay)
    n = 2

    rng = Random.MersenneTwister(42)
    _, welfares = run_sa_ensemble(Q, offset, n, 100, rng)
    # Optimal is all-cooperate: welfare = R * 1 = 3
    optimal = Float64(R)
    rate = mean(welfares .>= optimal - 1e-8)
    @test rate > 0.8
end

@testset "Exact quantum two-node PD" begin
    g = SimpleGraph(2)
    add_edge!(g, 1, 2)
    Q, offset = build_qubo(g, R, S, T, P_pay)
    n = 2

    rng = Random.MersenneTwister(123)
    config, welfare, _ = run_exact_quantum(Q, offset, n; Gamma=3.0, n_steps=50,
                                            n_samples=1000, rng=rng)
    @test welfare ≈ Float64(R)
end
