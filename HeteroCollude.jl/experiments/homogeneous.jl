# Experiments 1-3: homogeneous populations (the experimental controls).
#
#   julia --project=. experiments/homogeneous.jl

include(joinpath(@__DIR__, "common.jl"))

cfg = default_config()
println("Experiments 1-3: T=$(cfg.T), replications=$(cfg.n_reps), base seed=$(cfg.base_seed)")
println("Elevated threshold (highest symmetric stage-Nash price): ",
        highest_symmetric_nash_price(cfg.params))

runs, traj = run_homogeneous_experiments(cfg)
summary = summarize_runs(runs)

write_csv("exp1_3_runs.csv", runs)
write_csv("exp1_3_summary.csv", summary)
write_csv("exp1_3_trajectories.csv", traj)
print_summary(summary, HEADLINE_METRICS)
