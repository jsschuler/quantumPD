function exact_solve(Q::Matrix{Float64}, offset::Float64, n::Int)
    best_welfare = -Inf
    best_config = zeros(Int, n)
    for mask in 0:(2^n - 1)
        config = [(mask >> (i-1)) & 1 for i in 1:n]
        w = welfare_from_config(config, Q, offset)
        if w > best_welfare || (w == best_welfare && sum(config) < sum(best_config))
            best_welfare = w
            best_config = config
        end
    end
    return best_config, best_welfare
end
