# qlearning.jl
# Agent A: tabular Q-learning with epsilon-greedy exploration.
#
# State  s_t = (own previous action, lowest rival previous action)  -> K^2 states
# Action a_t = price index in 1:K
# Reward r_t = own period profit
# Update Q(s,a) <- Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
#
# Each agent maximises its OWN profit; there is no shared reward.

# ---------------------------------------------------------------------------
# Exploration schedules.  t is the period number, starting at 1.
# ---------------------------------------------------------------------------
abstract type EpsilonSchedule end

"ε_t = ε for all t."
struct ConstantEpsilon <: EpsilonSchedule
    ε::Float64
end

"Linear decay from ε0 (t=1) to ε_end (t=T), constant afterwards."
struct LinearDecay <: EpsilonSchedule
    ε0::Float64
    ε_end::Float64
    T::Int
    LinearDecay(ε0, ε_end, T) = T > 1 ? new(ε0, ε_end, T) :
        throw(ArgumentError("LinearDecay needs T > 1"))
end

"ε_t = max(ε_min, ε0 * rate^(t-1))."
struct ExponentialDecay <: EpsilonSchedule
    ε0::Float64
    rate::Float64
    ε_min::Float64
end

epsilon(s::ConstantEpsilon, t::Integer) = s.ε
epsilon(s::LinearDecay, t::Integer) =
    s.ε0 + (s.ε_end - s.ε0) * clamp((t - 1) / (s.T - 1), 0.0, 1.0)
epsilon(s::ExponentialDecay, t::Integer) = max(s.ε_min, s.ε0 * s.rate^(t - 1))

# ---------------------------------------------------------------------------
# Initial Q-values
# ---------------------------------------------------------------------------

"""
    random_play_q0(params, γ) -> Vector{Float64}

Action-specific initial value: expected discounted profit if the agent plays
action `a` now and all rivals choose uniformly at random forever:
    Q0(a) = E_rivals[ π(a, a_-i) ] / (1 - γ).
Computed exactly by enumerating rival profiles (K^(n-1) of them).
Zero-initialisation is NOT neutral here: profits are >= 0, so the first action
to earn anything would become greedy by construction.
"""
function random_play_q0(p::MarketParams, γ::Real)
    0 <= γ < 1 || throw(ArgumentError("γ must be in [0,1)"))
    K, n = n_actions(p), p.n_firms
    rival_profiles = Iterators.product(ntuple(_ -> 1:K, n - 1)...)
    n_prof = K^(n - 1)
    q0 = zeros(K)
    for a in 1:K
        total = 0.0
        for rv in rival_profiles
            total += clear_market(p, [a; collect(rv)]).profits[1]
        end
        q0[a] = total / n_prof / (1 - γ)
    end
    return q0
end

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

"Integer id in 1:K^2 for the state (own previous action, lowest rival previous action)."
state_index(K::Integer, own::Integer, rival::Integer) = (own - 1) * K + rival

mutable struct QLearningAgent <: AbstractAgent
    K::Int
    α::Float64
    γ::Float64
    schedule::EpsilonSchedule
    q0::Vector{Float64}          # per-action initial values (kept for reset!)
    Q::Matrix{Float64}           # K^2 states × K actions
    state::Int                   # current state id
    last_action::Int             # action chosen in the current period
end

"""
    QLearningAgent(params; α=0.1, γ=0.95, schedule=LinearDecay(0.2, 0.01, DEFAULT_PERIODS),
                   q_init=:random_play, rng=Random.default_rng())

`q_init` is `:random_play` (default), `:zero`, or a number.
"""
function QLearningAgent(p::MarketParams; α::Real = 0.1, γ::Real = 0.95,
                        schedule::EpsilonSchedule = LinearDecay(0.2, 0.01, DEFAULT_PERIODS),
                        q_init = :random_play,
                        rng::AbstractRNG = Random.default_rng())
    0 < α <= 1  || throw(ArgumentError("α must be in (0,1]"))
    0 <= γ < 1  || throw(ArgumentError("γ must be in [0,1)"))
    K = n_actions(p)
    q0 = q_init === :random_play ? random_play_q0(p, γ) :
         q_init === :zero        ? zeros(K) :
         q_init isa Real         ? fill(Float64(q_init), K) :
         throw(ArgumentError("q_init must be :random_play, :zero, or a number"))
    agent = QLearningAgent(K, Float64(α), Float64(γ), schedule, q0,
                           repeat(q0', K^2, 1), 1, 0)
    reset!(agent, rng)
    return agent
end

"Reset Q-table and draw a random initial state (own action, rival action)."
function reset!(a::QLearningAgent, rng::AbstractRNG)
    a.Q .= a.q0'
    a.state = state_index(a.K, rand(rng, 1:a.K), rand(rng, 1:a.K))
    a.last_action = 0
    return a
end

"Greedy action in state `s`, random among exact ties."
greedy_action(a::QLearningAgent, s::Integer, rng::AbstractRNG) =
    argmax_random_ties(rng, @view a.Q[s, :])

function act!(a::QLearningAgent, rng::AbstractRNG, t::Integer)
    k = rand(rng) < epsilon(a.schedule, t) ? rand(rng, 1:a.K) :
                                              greedy_action(a, a.state, rng)
    a.last_action = k
    return k
end

function observe!(a::QLearningAgent, obs::Observation, t::Integer)
    obs.own_action == a.last_action ||
        throw(ArgumentError("observation does not match the action this agent chose"))
    s′     = state_index(a.K, obs.own_action, obs.min_rival_action)
    target = obs.own_profit + a.γ * maximum(@view a.Q[s′, :])
    a.Q[a.state, a.last_action] += a.α * (target - a.Q[a.state, a.last_action])
    a.state = s′
    return nothing
end
