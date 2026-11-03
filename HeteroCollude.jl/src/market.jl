# market.jl
# One-shot market clearing for the stage game. No learning, no memory.
#
# Terminology (used consistently throughout the project):
#   posted price       : p_i, the price firm i announces this period
#   transaction price  : P = min_i p_i. Goods are homogeneous, so every unit
#                        traded in the period trades at this single price.
#   market clearing    : quantity Q = D(P), split equally among firms posting P
#   firms posting above P sell nothing (zero profit)

"Full record of one period of market interaction."
struct MarketOutcome
    price_indices::Vector{Int}
    posted_prices::Vector{Float64}
    transaction_price::Float64
    quantity::Float64              # total quantity traded, Q = D(P)
    quantities::Vector{Float64}    # firm-level sales
    profits::Vector{Float64}       # (p_i - c) q_i
    consumer_surplus::Float64
    producer_surplus::Float64      # sum of firm profits
    total_welfare::Float64         # CS + PS
end

"""
    clear_market(params, idx)

Clear the market given each firm's chosen action index. Ties at the lowest
price split demand equally (fractional quantities are allowed).
Comparisons use integer indices, so ties are exact.
"""
function clear_market(p::MarketParams, idx::AbstractVector{<:Integer})
    length(idx) == p.n_firms ||
        throw(ArgumentError("expected $(p.n_firms) price choices, got $(length(idx))"))
    all(k -> 1 <= k <= n_actions(p), idx) ||
        throw(ArgumentError("action index out of range 1:$(n_actions(p))"))

    ids     = collect(Int, idx)
    posted  = p.prices[ids]
    k_low   = minimum(ids)
    P       = p.prices[k_low]
    Q       = demand(p, P)
    winners = ids .== k_low
    share   = Q / count(winners)
    q       = [w ? share : 0.0 for w in winners]
    profits = (posted .- p.c) .* q
    cs      = consumer_surplus(p, P)
    ps      = sum(profits)
    return MarketOutcome(ids, posted, P, Q, q, profits, cs, ps, cs + ps)
end

# ---------------------------------------------------------------------------
# Stage-game equilibrium checks (used to validate benchmarks on the discrete grid)
# ---------------------------------------------------------------------------

"""
    best_deviation_gain(params, idx, i)

Largest profit increase firm `i` can obtain by unilaterally switching to any
other grid price, holding rivals fixed. Returns 0.0 if no deviation helps.
"""
function best_deviation_gain(p::MarketParams, idx::AbstractVector{<:Integer}, i::Integer)
    trial = collect(Int, idx)
    base  = clear_market(p, trial).profits[i]
    best  = 0.0
    for k in 1:n_actions(p)
        trial[i] = k
        best = max(best, clear_market(p, trial).profits[i] - base)
    end
    return best
end

"True if no firm has a profitable unilateral deviation (weak Nash, tolerance `atol`)."
is_nash(p::MarketParams, idx::AbstractVector{<:Integer}; atol::Real = 1e-9) =
    all(i -> best_deviation_gain(p, idx, i) <= atol, 1:p.n_firms)
