# evolutionary.jl
# Agent B: evolutionary learning over a small population of reactive policies.
#
# Policy (genome, 4 integer genes, each a price index in 1:K):
#   gene 1: price after being UNDERCUT   (lowest rival price < my previous price)
#   gene 2: price after a TIE            (lowest rival price == my previous price)
#   gene 3: price after LEADING          (lowest rival price  > my previous price)
#   gene 4: opening price (first period only)
# This is deliberately a much coarser state space than the Q-learner's (3 vs 289).
#
# Evaluation: ONE agent, ONE stream of experience. Candidates take turns playing
# the real market for `eval_periods` periods each; fitness = own mean profit
# during that turn. After every candidate has had a turn (one generation):
#   Selection   : truncation, keep the `n_elite` best genomes unchanged (retention)
#   Replacement : remaining slots filled with mutated copies of random elites
#   Mutation    : each gene shifts w.p. `mut_rate` by a random step in ±1..mut_step
# All candidates are re-evaluated every generation because rivals' behaviour is
# non-stationary (fitness estimates are noisy by design).

const N_GENES = 4

"Coarse state from last period: 1 = undercut, 2 = tie, 3 = led (strictly lowest)."
relative_state(own::Integer, rival::Integer) = rival < own ? 1 : (rival == own ? 2 : 3)

mutable struct EvolutionaryAgent <: AbstractAgent
    K::Int
    pop_size::Int
    n_elite::Int
    eval_periods::Int
    mut_rate::Float64
    mut_step::Int
    pop::Matrix{Int}            # N_GENES × pop_size
    fitness::Vector{Float64}    # mean profit of each candidate's latest turn
    profit_sum::Float64
    candidate::Int              # candidate currently playing
    t_eval::Int                 # periods played in the current turn
    state::Int                  # 0 before first observation, else relative_state
    last_action::Int
    generation::Int
    rng::Xoshiro                # own RNG for mutation/selection, seeded in reset!
end

function EvolutionaryAgent(p::MarketParams; pop_size::Integer = 10, n_elite::Integer = 5,
                           eval_periods::Integer = 20, mut_rate::Real = 0.25,
                           mut_step::Integer = 3, rng::AbstractRNG = Random.default_rng())
    pop_size >= 2                  || throw(ArgumentError("pop_size must be >= 2"))
    1 <= n_elite < pop_size        || throw(ArgumentError("need 1 <= n_elite < pop_size"))
    eval_periods >= 1              || throw(ArgumentError("eval_periods must be >= 1"))
    0 <= mut_rate <= 1             || throw(ArgumentError("mut_rate must be in [0,1]"))
    mut_step >= 1                  || throw(ArgumentError("mut_step must be >= 1"))
    ag = EvolutionaryAgent(n_actions(p), pop_size, n_elite, eval_periods,
                           Float64(mut_rate), mut_step, ones(Int, N_GENES, pop_size),
                           zeros(pop_size), 0.0, 1, 0, 0, 0, 0, Xoshiro(0))
    reset!(ag, rng)
    return ag
end

function reset!(a::EvolutionaryAgent, rng::AbstractRNG)
    a.rng = Xoshiro(rand(rng, UInt64))
    a.pop = rand(a.rng, 1:a.K, N_GENES, a.pop_size)
    fill!(a.fitness, 0.0)
    a.profit_sum = 0.0
    a.candidate, a.t_eval, a.state, a.last_action, a.generation = 1, 0, 0, 0, 0
    return a
end

function act!(a::EvolutionaryAgent, rng::AbstractRNG, t::Integer)
    g = @view a.pop[:, a.candidate]
    k = a.state == 0 ? g[4] : g[a.state]
    a.last_action = k
    return k
end

function observe!(a::EvolutionaryAgent, obs::Observation, t::Integer)
    obs.own_action == a.last_action ||
        throw(ArgumentError("observation does not match the action this agent chose"))
    a.profit_sum += obs.own_profit
    a.t_eval += 1
    a.state = relative_state(obs.own_action, obs.min_rival_action)
    if a.t_eval == a.eval_periods
        a.fitness[a.candidate] = a.profit_sum / a.eval_periods
        a.profit_sum, a.t_eval = 0.0, 0
        a.candidate += 1
        if a.candidate > a.pop_size
            evolve!(a)
            a.candidate = 1
        end
    end
    return nothing
end

"Mutated copy of a genome."
function mutate(a::EvolutionaryAgent, genome::AbstractVector{<:Integer})
    g = collect(Int, genome)
    for i in eachindex(g)
        if rand(a.rng) < a.mut_rate
            step = rand(a.rng, 1:a.mut_step) * (rand(a.rng, Bool) ? 1 : -1)
            g[i] = clamp(g[i] + step, 1, a.K)
        end
    end
    return g
end

"""
    evolve!(agent)

One generation step: rank by fitness (ties broken at random), keep the `n_elite`
best genomes in columns 1:n_elite, fill the rest with mutated copies of random elites.
"""
function evolve!(a::EvolutionaryAgent)
    perm   = randperm(a.rng, a.pop_size)
    order  = perm[sortperm(a.fitness[perm]; rev = true, alg = Base.Sort.MergeSort)]
    elites = order[1:a.n_elite]
    newpop = similar(a.pop)
    for (j, e) in enumerate(elites)
        newpop[:, j] = a.pop[:, e]
    end
    for j in (a.n_elite + 1):a.pop_size
        parent = rand(a.rng, elites)
        newpop[:, j] = mutate(a, @view a.pop[:, parent])
    end
    a.pop = newpop
    fill!(a.fitness, 0.0)
    a.generation += 1
    return a
end
