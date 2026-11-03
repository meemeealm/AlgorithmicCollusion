# simulation.jl
# Repeated-interaction engine: population specs, configuration, seeds, histories.
#
# One period t:
#   1. every agent chooses an action simultaneously (act!)
#   2. the market clears (clear_market)
#   3. every agent receives its own Observation and learns (observe!)
# Nothing else is shared between agents: no communication, no common state.

# ---------------------------------------------------------------------------
# Population specification
# ---------------------------------------------------------------------------

const AGENT_KINDS = (:q, :evo, :rule)

"""
    Population(kinds; name=nothing, q_kwargs=(;), evo_kwargs=(;), rule_ranges=RuleParamRanges())

Which agent occupies each firm slot (`:q`, `:evo`, `:rule`) and how each kind is
configured. Rule-based parameters are drawn independently per agent from
`rule_ranges` (see `fixed_rule_ranges` for the identical, noise-free ablation).
"""
struct Population
    name::String
    kinds::Vector{Symbol}
    q_kwargs::NamedTuple
    evo_kwargs::NamedTuple
    rule_ranges::RuleParamRanges
end

function _auto_name(kinds)
    parts = String[]
    for (sym, tag) in zip(AGENT_KINDS, ("Q", "E", "R"))
        c = count(==(sym), kinds)
        c > 0 && push!(parts, "$(c)$(tag)")
    end
    return join(parts, "+")
end

function Population(kinds::AbstractVector{Symbol}; name = nothing, q_kwargs::NamedTuple = (;),
                    evo_kwargs::NamedTuple = (;), rule_ranges::RuleParamRanges = RuleParamRanges())
    all(k -> k in AGENT_KINDS, kinds) ||
        throw(ArgumentError("agent kinds must be among $(AGENT_KINDS)"))
    length(kinds) >= 2 || throw(ArgumentError("a population needs at least 2 agents"))
    nm = name === nothing ? _auto_name(kinds) : String(name)
    return Population(nm, collect(Symbol, kinds), q_kwargs, evo_kwargs, rule_ranges)
end

"Experiments 1-3: `n` agents of one kind."
homogeneous_population(kind::Symbol, n::Integer = 3; kwargs...) =
    Population(fill(kind, n); kwargs...)

"Experiment 4: one agent of each type."
mixed_population(; kwargs...) = Population([:q, :evo, :rule]; kwargs...)

"Experiment 5: a population with the given number of each agent type."
population_from_counts(q::Integer, evo::Integer, rule::Integer; kwargs...) =
    Population(vcat(fill(:q, q), fill(:evo, evo), fill(:rule, rule)); kwargs...)

"All (n_q, n_evo, n_rule) compositions with n_q + n_evo + n_rule == n."
population_compositions(n::Integer) =
    [(q, e, n - q - e) for q in n:-1:0 for e in (n - q):-1:0]

"Rule-parameter ranges collapsed to single values: identical agents (default: noise-free)."
fixed_rule_ranges(; step::Integer = 1, cut_step::Integer = 2, noise::Real = 0.0,
                    noise_width::Integer = 2) =
    RuleParamRanges(step = step:step, cut_step = cut_step:cut_step,
                    noise = (noise, noise), noise_width = noise_width:noise_width)

"""
    build_agent(kind, population, params, T, rng)

Construct one agent. Unless `q_kwargs` supplies a `schedule`, the Q-learner's
ε-decay horizon is set to the run length `T`, so the exploration schedule always
spans the whole run.
"""
function build_agent(kind::Symbol, pop::Population, p::MarketParams, T::Integer, rng::AbstractRNG)
    if kind === :q
        kw = pop.q_kwargs
        return haskey(kw, :schedule) ? QLearningAgent(p; kw..., rng = rng) :
               QLearningAgent(p; kw..., schedule = LinearDecay(0.2, 0.01, T), rng = rng)
    elseif kind === :evo
        return EvolutionaryAgent(p; pop.evo_kwargs..., rng = rng)
    elseif kind === :rule
        return sample_rule_agent(p, rng; ranges = pop.rule_ranges)
    end
    throw(ArgumentError("unknown agent kind $kind"))
end

# ---------------------------------------------------------------------------
# Agent descriptions (recorded with every history)
# ---------------------------------------------------------------------------
agent_label(::QLearningAgent)     = "Q-learning"
agent_label(::EvolutionaryAgent)  = "Evolutionary"
agent_label(::AdaptiveRuleAgent)  = "Rule-based"
agent_label(::AbstractAgent)      = "Other"

agent_params(a::QLearningAgent) = Dict{Symbol,Float64}(:alpha => a.α, :gamma => a.γ)
agent_params(a::EvolutionaryAgent) = Dict{Symbol,Float64}(
    :pop_size => a.pop_size, :n_elite => a.n_elite, :eval_periods => a.eval_periods,
    :mut_rate => a.mut_rate, :mut_step => a.mut_step, :generations => a.generation)
agent_params(a::AdaptiveRuleAgent) = Dict{Symbol,Float64}(
    :step => a.step, :cut_step => a.cut_step, :noise => a.noise, :noise_width => a.noise_width)
agent_params(::AbstractAgent) = Dict{Symbol,Float64}()

# ---------------------------------------------------------------------------
# Configuration and seeds
# ---------------------------------------------------------------------------

"""
    SimulationConfig(; params=MarketParams(), T=DEFAULT_PERIODS,
                       n_reps=DEFAULT_REPLICATIONS, base_seed=2026)
"""
struct SimulationConfig
    params::MarketParams
    T::Int
    n_reps::Int
    base_seed::UInt64
end

function SimulationConfig(; params::MarketParams = MarketParams(), T::Integer = DEFAULT_PERIODS,
                            n_reps::Integer = DEFAULT_REPLICATIONS, base_seed::Integer = 2026)
    T >= 2     || throw(ArgumentError("T must be >= 2"))
    n_reps >= 1 || throw(ArgumentError("n_reps must be >= 1"))
    return SimulationConfig(params, Int(T), Int(n_reps), UInt64(base_seed))
end

"""
    replication_seeds(base_seed, n)

`n` independent seeds drawn from a master stream. Adding replications later
leaves earlier seeds unchanged (the first `k` seeds do not depend on `n`).
All populations use the same seed list, so replication r is comparable by label
across experiments.
"""
replication_seeds(base_seed::Integer, n::Integer) = rand(Xoshiro(base_seed), UInt64, n)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

"Complete record of one simulation run (rows = periods, columns = firms)."
struct SimulationHistory
    population::String
    seed::UInt64
    T::Int
    n_firms::Int
    agent_labels::Vector{String}
    agent_params::Vector{Dict{Symbol,Float64}}   # recorded at the end of the run
    price_indices::Matrix{Int}
    posted_prices::Matrix{Float64}
    transaction_price::Vector{Float64}
    quantity::Vector{Float64}
    quantities::Matrix{Float64}
    profits::Matrix{Float64}
    consumer_surplus::Vector{Float64}
    producer_surplus::Vector{Float64}
    total_welfare::Vector{Float64}
end

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

"""
    run_simulation_with_state(config, population, seed) -> (history, agents, rng)

Run one replication and also return the agents and RNG in their final state (used by diagnostics). Everything random (agent construction, exploration,
mutation, noise) derives from `Xoshiro(seed)`, so a (config, population, seed)
triple reproduces its history exactly.
"""
function run_simulation_with_state(cfg::SimulationConfig, pop::Population, seed::Integer)
    p, T = cfg.params, cfg.T
    n = p.n_firms
    length(pop.kinds) == n ||
        throw(ArgumentError("population has $(length(pop.kinds)) agents but the market has $n firms"))

    rng    = Xoshiro(seed)
    agents = AbstractAgent[build_agent(k, pop, p, T, rng) for k in pop.kinds]   # constructors reset agents

    price_indices = zeros(Int, T, n)
    posted        = zeros(T, n)
    P, Q, CS, PS, W = (zeros(T) for _ in 1:5)
    quantities    = zeros(T, n)
    profits       = zeros(T, n)
    idx           = zeros(Int, n)

    for t in 1:T
        for i in 1:n
            idx[i] = act!(agents[i], rng, t)
        end
        o = clear_market(p, idx)
        for i in 1:n
            observe!(agents[i], make_observation(p, o, i), t)
        end
        price_indices[t, :] = o.price_indices
        posted[t, :]        = o.posted_prices
        P[t], Q[t]          = o.transaction_price, o.quantity
        quantities[t, :]    = o.quantities
        profits[t, :]       = o.profits
        CS[t], PS[t], W[t]  = o.consumer_surplus, o.producer_surplus, o.total_welfare
    end

    hist = SimulationHistory(pop.name, UInt64(seed), T, n,
                             [agent_label(a) for a in agents], [agent_params(a) for a in agents],
                             price_indices, posted, P, Q, quantities, profits, CS, PS, W)
    return hist, agents, rng
end

"""
    run_simulation(config, population, seed) -> SimulationHistory

One replication. Everything random derives from `Xoshiro(seed)`, so a
(config, population, seed) triple reproduces its history exactly.
"""
run_simulation(cfg::SimulationConfig, pop::Population, seed::Integer) =
    first(run_simulation_with_state(cfg, pop, seed))

"""
    run_replications(config, population; seeds=replication_seeds(config.base_seed, config.n_reps))

Independent replications, one history per seed.
"""
run_replications(cfg::SimulationConfig, pop::Population;
                 seeds::AbstractVector{<:Integer} = replication_seeds(cfg.base_seed, cfg.n_reps)) =
    [run_simulation(cfg, pop, s) for s in seeds]
