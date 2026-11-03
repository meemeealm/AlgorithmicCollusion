# agents.jl
# Common agent interface and the information every agent receives.
#
# Information condition (identical for ALL agent types, all experiments):
#   after each period agent i observes
#     - its own action (price index)
#     - its own profit and quantity sold (i.e. whether it captured demand)
#     - the LOWEST price index posted by its rivals
#   It does not observe rivals' individual prices, rivals' profits, or any
#   communication channel.
#
# Per-period protocol (driven by the simulation engine, Phase 4):
#   reset!(agent, rng)            once per replication
#   k = act!(agent, rng, t)       choose action index for period t
#   observe!(agent, obs, t)       learn from the realised outcome of period t

abstract type AbstractAgent end

"What a single agent sees after a period."
struct Observation
    own_action::Int
    own_profit::Float64
    own_quantity::Float64
    min_rival_action::Int
end

"Build agent `i`'s observation from a cleared market."
function make_observation(::MarketParams, o::MarketOutcome, i::Integer)
    min_rival = minimum(o.price_indices[j] for j in eachindex(o.price_indices) if j != i)
    return Observation(o.price_indices[i], o.profits[i], o.quantities[i], min_rival)
end

"""
    act!(agent, rng, t) -> Int
Choose an action index (1-based) for period `t`.
"""
function act! end

"""
    observe!(agent, obs, t)
Update the agent after period `t`.
"""
function observe! end

"""
    reset!(agent, rng)
Restore the agent to its initial condition at the start of a replication.
"""
function reset! end

"Index of a maximal element, breaking exact ties uniformly at random."
function argmax_random_ties(rng::AbstractRNG, v::AbstractVector)
    m = maximum(v)
    return rand(rng, findall(==(m), v))
end

"Non-learning agent that always posts the same price. Used as a test fixture and baseline."
struct FixedPriceAgent <: AbstractAgent
    action::Int
end
act!(a::FixedPriceAgent, rng::AbstractRNG, t::Integer) = a.action
observe!(::FixedPriceAgent, ::Observation, ::Integer) = nothing
reset!(::FixedPriceAgent, ::AbstractRNG) = nothing
