# Experiment 4: one agent of each learning type, compared with the homogeneous controls.
#
#   julia --project=. experiments/heterogeneous.jl

include(joinpath(@__DIR__, "common.jl"))

cfg = default_config()
println("Experiment 4: T=$(cfg.T), replications=$(cfg.n_reps), base seed=$(cfg.base_seed)")

println("Controls:")
hom_runs, hom_traj = run_homogeneous_experiments(cfg)
println("Heterogeneous population:")
het_runs, het_traj = evaluate_population(cfg, mixed_population(); experiment = "exp4_mixed")

runs = vcat(hom_runs, het_runs; cols = :union)
traj = vcat(hom_traj, het_traj; cols = :union)
summary = summarize_runs(runs)

# Mixed vs each homogeneous control: permutation tests on the headline outcomes.
tested = [:price_final, :welfare_ratio_final, :ps_final, :elevated_share_final]
controls = unique(hom_runs.population)
rows = NamedTuple[]
for m in tested, c in controls
    x = Float64.(het_runs[!, m])
    y = Float64.(hom_runs[hom_runs.population .== c, m])
    t = permutation_test(x, y; B = 10_000, rng = Xoshiro(7))
    push!(rows, (; metric = String(m), mixed = "1Q+1E+1R", control = c,
                   mixed_mean = mean(x), control_mean = mean(y), diff = t.diff, p_value = t.p_value))
end
comparisons = DataFrame(rows)
comparisons.p_bonferroni = min.(1.0, comparisons.p_value .* nrow(comparisons))

# Descriptive heterogeneity effect: mixed vs the average of the three homogeneous means.
for m in tested
    het  = mean(het_runs[!, m])
    ctrl = mean(mean(hom_runs[hom_runs.population .== c, m]) for c in controls)
    println("\n", m, ": mixed = ", round(het; digits = 3),
            "   mean of homogeneous controls = ", round(ctrl; digits = 3))
end

write_csv("exp4_runs.csv", runs)
write_csv("exp4_summary.csv", summary)
write_csv("exp4_trajectories.csv", traj)
write_csv("exp4_comparisons.csv", comparisons)
print_summary(filter(r -> r.population == "1Q+1E+1R", summary),
              vcat(HEADLINE_METRICS, [:profit_firm1, :profit_firm2, :profit_firm3]))
println("\nIn the mixed population, firm1 = Q-learning, firm2 = Evolutionary, firm3 = Rule-based.")
println("\nMixed vs controls (Bonferroni over $(nrow(comparisons)) tests):")
show(comparisons; allrows = true, allcols = true); println()
