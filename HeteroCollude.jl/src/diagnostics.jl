# diagnostics.jl
# Deviation probe: does the market react to an undercut the way a conditional
# (punishing) strategy would, or only the way independent adaptation would?
#
# Procedure, applied to the FINAL state of a finished run:
#   1. copy agents and RNG -> "baseline" and "deviation" continuations
#   2. in the deviation copy only, force firm 1 to undercut the last period's lowest
#      price by one grid step for ONE period
#   3. let both copies continue for `horizon` periods with learning switched on
#   4. gap[h] = (rivals' mean posted price, deviation) - (same, baseline)
#
# Reading it: a negative gap right after the deviation shows rivals cut prices. That
# alone is also what a myopic profit-maximiser does, so it is NOT evidence of collusion.
# What distinguishes a punishment strategy is REVERSION: gap climbing back toward 0
# afterwards (`reversion` > 0). The two continuations share a starting RNG state, but their
# random streams drift apart once behaviour differs, so single-run gaps are noisy;
# interpret averages across replications.

"Make an agent record `k` as the action it just took (so learning credits the forced action)."
force_action!(a::AbstractAgent, k::Integer) = (a.last_action = k; nothing)
force_action!(::FixedPriceAgent, ::Integer) = nothing

function _continue!(agents, rng, p::MarketParams, t0::Integer, horizon::Integer, forced)
    n = length(agents)
    prices = zeros(horizon, n)
    idx = zeros(Int, n)
    for s in 1:horizon
        t = t0 + s
        for i in 1:n
            idx[i] = act!(agents[i], rng, t)
        end
        if s == 1 && forced !== nothing
            idx[1] = forced
            force_action!(agents[1], forced)
        end
        o = clear_market(p, idx)
        for i in 1:n
            observe!(agents[i], make_observation(p, o, i), t)
        end
        prices[s, :] = o.posted_prices
    end
    return prices
end

"""
    deviation_probe(agents, rng, params, history; horizon=10)

Run the probe described above. `agents` and `rng` are consumed (use the values
returned by `run_simulation_with_state`). If the market is already at the lowest
grid price no undercut exists and `possible=false` with NaN results.
"""
function deviation_probe(agents, rng::AbstractRNG, p::MarketParams, h::SimulationHistory;
                         horizon::Integer = 10)
    last_low = minimum(h.price_indices[end, :])
    if last_low <= 1
        return (; possible = false, gap = fill(NaN, horizon), depth = NaN,
                  depth_period = 0, final_gap = NaN, reversion = NaN)
    end
    forced = last_low - 1
    base = _continue!(deepcopy(agents), copy(rng), p, h.T, horizon, nothing)
    dev  = _continue!(agents, rng, p, h.T, horizon, forced)
    rivals = 2:p.n_firms
    gap = vec(mean(dev[:, rivals]; dims = 2)) .- vec(mean(base[:, rivals]; dims = 2))
    depth, per = findmin(gap)
    return (; possible = true, gap, depth, depth_period = per,
              final_gap = gap[end], reversion = gap[end] - depth)
end
