# Sanity check only: Experiments 1-4 at default settings, few replications.
# This is NOT the Phase 5 analysis and says nothing about collusion by itself.
#
#   julia --project=. experiments/smoke_run.jl

using HeteroCollude, Statistics

cfg  = SimulationConfig(n_reps = 5)
pops = [homogeneous_population(:q), homogeneous_population(:evo),
        homogeneous_population(:rule), mixed_population()]
w    = (cfg.T - min(1000, cfg.T) + 1):cfg.T          # late window

println("population   mean price   sd across reps   CI      mean welfare")
for pop in pops
    hs     = run_replications(cfg, pop)
    prices = [mean(h.transaction_price[w]) for h in hs]
    welf   = [mean(h.total_welfare[w]) for h in hs]
    println(rpad(pop.name, 13), rpad(round(mean(prices); digits = 2), 13),
            rpad(round(std(prices); digits = 2), 17),
            rpad(round(collusion_index(mean(prices), cfg.params); digits = 3), 8),
            round(mean(welf); digits = 1))
end
