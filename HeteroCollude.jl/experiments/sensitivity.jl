# Experiment 5: population composition and robustness.
#   A. all 10 compositions of Q / evolutionary / rule-based agents (3 firms)
#   B. rule-based agents: identical vs diverse parameters x noise on/off (drift ablation)
#   C. Q-learning: learning rate and discount factor
#   D. longer horizon (3x) for Q-only and mixed populations (is Q-learning under-trained?)
#   E. finer price grid (step 2.5) for the mixed population
#
#   julia --project=. experiments/sensitivity.jl

include(joinpath(@__DIR__, "common.jl"))

cfg = default_config()
println("Experiment 5: T=$(cfg.T), replications=$(cfg.n_reps), base seed=$(cfg.base_seed)")

runs, trajs = DataFrame[], DataFrame[]
function add!(cfg_, pop; experiment, label, kwargs...)
    println("  ", experiment, ": ", label)
    r, t = evaluate_population(cfg_, pop; experiment = experiment, label = label, kwargs...)
    push!(runs, r); push!(trajs, t)
end

# A. composition
for (q, e, r) in population_compositions(cfg.params.n_firms)
    pop = population_from_counts(q, e, r)
    add!(cfg, pop; experiment = "exp5a_composition", label = pop.name)
end

# B. rule-based drift ablation
rule_variants = [
    ("rule: identical, no noise", fixed_rule_ranges(noise = 0.0)),
    ("rule: identical, noise",    fixed_rule_ranges(noise = 0.05)),
    ("rule: diverse, no noise",   RuleParamRanges(noise = (0.0, 0.0))),
    ("rule: diverse, noise",      RuleParamRanges())]
for (label, ranges) in rule_variants
    add!(cfg, homogeneous_population(:rule; rule_ranges = ranges); experiment = "exp5b_rule_ablation", label = label)
end

# C. Q-learning hyperparameters (one at a time)
for (label, kw) in [("Q: alpha=0.05", (α = 0.05,)), ("Q: alpha=0.2", (α = 0.2,)),
                    ("Q: gamma=0.9", (γ = 0.9,)),   ("Q: gamma=0.99", (γ = 0.99,))]
    add!(cfg, homogeneous_population(:q; q_kwargs = kw); experiment = "exp5c_q_params", label = label)
end

# D. horizon (the Q epsilon schedule follows T automatically)
long = SimulationConfig(params = cfg.params, T = 3 * cfg.T, n_reps = cfg.n_reps, base_seed = cfg.base_seed)
add!(long, homogeneous_population(:q); experiment = "exp5d_horizon", label = "3Q T=$(long.T)")
add!(long, mixed_population();          experiment = "exp5d_horizon", label = "1Q+1E+1R T=$(long.T)")

# E. finer price grid
fine = SimulationConfig(params = MarketParams(p_step = 2.5), T = cfg.T, n_reps = cfg.n_reps, base_seed = cfg.base_seed)
add!(fine, mixed_population(); experiment = "exp5e_grid", label = "1Q+1E+1R step=2.5")

all_runs = reduce((a, b) -> vcat(a, b; cols = :union), runs)
all_traj = reduce((a, b) -> vcat(a, b; cols = :union), trajs)
summary  = summarize_runs(all_runs)

write_csv("exp5_runs.csv", all_runs)
write_csv("exp5_summary.csv", summary)
write_csv("exp5_trajectories.csv", all_traj)
print_summary(summary, [:price_final, :ci_final, :elevated_share_final, :persistent,
                        :welfare_ratio_final, :all_equal_share_final, :dev_reversion])
println("\nNote: for the step=2.5 grid the elevated threshold and CI use that grid's own benchmarks.")
