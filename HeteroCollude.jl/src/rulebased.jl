# rulebased.jl
# Agent C: adaptive rule-based pricing ("performance-driven hill climbing"),
# with random noise and per-agent parameter diversity.
#
# Rule (uses only own previous price, own previous profit, whether demand was captured):
#   lost the market (quantity == 0) : cut price by `cut_step` grid steps, direction := down
#   captured, profit <= 0           : direction := up   (earning nothing: try a higher price)
#   captured, profit < previous     : reverse direction (last move hurt)
#   captured, profit >= previous    : keep direction    (last move did not hurt)
#   then move `step` grid steps in the current direction (clamped to the grid)
#
# Why noise and diversity: identical deterministic rules respond identically to the
# same market signals, so rule-based agents can move up or down in lockstep. That is
# synchronised adaptation to common conditions, not coordination, and it would be
# indistinguishable from collusion in the price data. Two countermeasures:
#   Noise     : with probability `noise` per period the POSTED price is the rule's price
#               shifted by a random nonzero amount of up to `noise_width` grid steps.
#               The rule then reacts to the price actually posted, so noise is part of
#               the dynamics, not measurement error.
#   Diversity : `step`, `cut_step`, `noise`, `noise_width` can differ across agents,
#               drawn once per agent from `RuleParamRanges` (sample_rule_agent).

mutable struct AdaptiveRuleAgent <: AbstractAgent
    K::Int
    step::Int
    cut_step::Int
    noise::Float64
    noise_width::Int
    k::Int                 # price index the rule intends to post next
    dir::Int               # +1 or -1
    prev_profit::Float64
    last_action::Int       # price index actually posted (intent plus noise)
end

function AdaptiveRuleAgent(p::MarketParams; step::Integer = 1, cut_step::Integer = 2,
                           noise::Real = 0.05, noise_width::Integer = 2,
                           rng::AbstractRNG = Random.default_rng())
    step >= 1 && cut_step >= 1 || throw(ArgumentError("step and cut_step must be >= 1"))
    0 <= noise <= 1            || throw(ArgumentError("noise must be in [0,1]"))
    noise_width >= 1           || throw(ArgumentError("noise_width must be >= 1"))
    ag = AdaptiveRuleAgent(n_actions(p), step, cut_step, Float64(noise), noise_width,
                           1, 1, -Inf, 0)
    reset!(ag, rng)
    return ag
end

"Restart the rule: random initial price and direction. Parameters are kept."
function reset!(a::AdaptiveRuleAgent, rng::AbstractRNG)
    a.k = rand(rng, 1:a.K)
    a.dir = rand(rng, (-1, 1))
    a.prev_profit = -Inf
    a.last_action = 0
    return a
end

function act!(a::AdaptiveRuleAgent, rng::AbstractRNG, t::Integer)
    k = a.k
    if a.noise > 0 && rand(rng) < a.noise
        shift = rand(rng, 1:a.noise_width) * (rand(rng, Bool) ? 1 : -1)
        k = clamp(k + shift, 1, a.K)
    end
    a.last_action = k
    return k
end

function observe!(a::AdaptiveRuleAgent, obs::Observation, t::Integer)
    obs.own_action == a.last_action ||
        throw(ArgumentError("observation does not match the action this agent chose"))
    if obs.own_quantity <= 0
        a.dir = -1
        a.k = clamp(obs.own_action - a.cut_step, 1, a.K)
    else
        if obs.own_profit <= 0
            a.dir = +1
        elseif obs.own_profit < a.prev_profit
            a.dir = -a.dir
        end
        a.k = clamp(obs.own_action + a.dir * a.step, 1, a.K)
    end
    a.prev_profit = obs.own_profit
    return nothing
end

# ---------------------------------------------------------------------------
# Parameter diversity
# ---------------------------------------------------------------------------

"""
    RuleParamRanges(; step=1:2, cut_step=1:3, noise=(0.02, 0.10), noise_width=1:3)

Ranges from which each rule-based agent's parameters are drawn independently
(integers uniformly, noise level uniformly on the interval). Collapse a range to a
single value (e.g. `step=1:1`, `noise=(0.05, 0.05)`) to switch diversity off.
"""
struct RuleParamRanges
    step::UnitRange{Int}
    cut_step::UnitRange{Int}
    noise::Tuple{Float64,Float64}
    noise_width::UnitRange{Int}
end

RuleParamRanges(; step = 1:2, cut_step = 1:3, noise = (0.02, 0.10), noise_width = 1:3) =
    RuleParamRanges(step, cut_step, (Float64(noise[1]), Float64(noise[2])), noise_width)

"One rule-based agent with parameters drawn from `ranges` using `rng`."
function sample_rule_agent(p::MarketParams, rng::AbstractRNG;
                           ranges::RuleParamRanges = RuleParamRanges())
    lo, hi = ranges.noise
    lo <= hi || throw(ArgumentError("noise range must satisfy lo <= hi"))
    step        = rand(rng, ranges.step)
    cut_step    = rand(rng, ranges.cut_step)
    noise       = lo + (hi - lo) * rand(rng)
    noise_width = rand(rng, ranges.noise_width)
    return AdaptiveRuleAgent(p; step = step, cut_step = cut_step, noise = noise,
                             noise_width = noise_width, rng = rng)
end

"`n` rule-based agents with independently drawn parameters."
diverse_rule_agents(p::MarketParams, n::Integer, rng::AbstractRNG;
                    ranges::RuleParamRanges = RuleParamRanges()) =
    [sample_rule_agent(p, rng; ranges = ranges) for _ in 1:n]
