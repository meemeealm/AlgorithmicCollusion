# diagnostics.jl
# Deviation probe: how do rivals' prices respond to a forced undercut?
#
# Procedure, applied to the FINAL state of a finished run:
#   1. copy agents and RNG into three continuations
#        base    : no deviation
#        dev     : firm 1 is forced to undercut last period's lowest price by one grid
#                  step for ONE period (shares base's RNG state)
#        placebo : no deviation, but a different random stream
#   2. all three continue for `horizon` periods with learning on
#   3. gap[s]         = rivals' mean posted price (dev)     - (base)
#      placebo_gap[s] = rivals' mean posted price (placebo) - (base)
#
# Summary statistics are WINDOW MEANS, never a minimum or maximum of the noisy gap
# (those are biased: the minimum of noise is negative even when nothing happens).
#   gap_early  = mean(gap[early])            default periods 2:4 (rivals react from period 2)
#   gap_late   = mean(gap[late])             default the last three periods
#   reversion  = gap_late - gap_early        > 0 means rivals' prices move back toward baseline
# The placebo versions (plc_*) show what the same statistics look like with NO deviation,
# so they should be centred near 0; they are an upper bound on the noise in the dev
# statistics (dev and base share a random stream, placebo and base do not).
#
# Reading it: a negative early gap shows rivals cut after the undercut. A myopic
# profit-maximiser does that too, and also drifts back afterwards, so neither a cut nor
# a recovery is evidence of collusion on its own. The probe is a descriptive comparison
# across populations (and against the placebo), not a collusion test.

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
    deviation_probe(agents, rng, params, history; horizon=10, early=2:4, late=(horizon-2):horizon)

Run the probe described above. `agents` and `rng` are consumed (use the values returned
by `run_simulation_with_state`). If the market is already at the lowest grid price no
undercut exists and `possible=false` with NaN results.
"""
function deviation_probe(agents, rng::AbstractRNG, p::MarketParams, h::SimulationHistory;
                         horizon::Integer = 10, early = 2:4, late = (horizon - 2):horizon)
    horizon >= 5 || throw(ArgumentError("horizon must be >= 5"))
    last_low = minimum(h.price_indices[end, :])
    if last_low <= 1
        nan = fill(NaN, horizon)
        return (; possible = false, gap = nan, placebo_gap = copy(nan), gap_early = NaN,
                  gap_late = NaN, reversion = NaN, plc_gap_early = NaN, plc_gap_late = NaN,
                  plc_reversion = NaN)
    end
    forced   = last_low - 1
    ctrl_rng = Xoshiro(rand(copy(rng), UInt64))
    ctrl = _continue!(deepcopy(agents), ctrl_rng, p, h.T, horizon, nothing)
    base = _continue!(deepcopy(agents), copy(rng), p, h.T, horizon, nothing)
    dev  = _continue!(agents, rng, p, h.T, horizon, forced)
    rival(m) = vec(mean(m[:, 2:p.n_firms]; dims = 2))
    gap  = rival(dev) .- rival(base)
    pgap = rival(ctrl) .- rival(base)
    ge, gl = mean(gap[early]), mean(gap[late])
    pe, pl = mean(pgap[early]), mean(pgap[late])
    return (; possible = true, gap, placebo_gap = pgap, gap_early = ge, gap_late = gl,
              reversion = gl - ge, plc_gap_early = pe, plc_gap_late = pl, plc_reversion = pl - pe)
end