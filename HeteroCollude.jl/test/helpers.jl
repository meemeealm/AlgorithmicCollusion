# Minimal repeated-game loop for tests (the full engine arrives in Phase 4).
# `reset=false` lets a test start agents from a hand-set internal state.
function play(p, agents, rng, T; reset::Bool = true)
    reset && foreach(a -> reset!(a, rng), agents)
    profits = zeros(T, length(agents))
    for t in 1:T
        idx = [act!(a, rng, t) for a in agents]
        o = clear_market(p, idx)
        for (i, a) in enumerate(agents)
            observe!(a, make_observation(p, o, i), t)
        end
        profits[t, :] = o.profits
    end
    return profits
end
