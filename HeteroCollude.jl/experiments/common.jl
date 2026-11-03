# Shared utilities for the experiment scripts.
#
# One-time setup (DataFrames is used only by these scripts, not by the package):
#   julia --project=. -e 'using Pkg; Pkg.add("DataFrames")'
#
# Quick, low-cost trial of any script (4 replications, 2,000 periods):
#   HETERO_QUICK=1 julia --project=. experiments/homogeneous.jl

using HeteroCollude, Statistics, Random, DataFrames

const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))

"Default configuration: 3 firms, 10,000 periods, 30 replications (or the quick variant)."
function default_config(; params::MarketParams = MarketParams())
    get(ENV, "HETERO_QUICK", "0") == "1" ?
        SimulationConfig(params = params, T = 2000, n_reps = 4) :
        SimulationConfig(params = params)
end

# ---- CSV output (no extra dependency) ---------------------------------------------
_fmt(x::AbstractFloat) = isnan(x) ? "NA" : string(x)
_fmt(x::AbstractString) = occursin(',', x) ? "\"" * x * "\"" : String(x)
_fmt(x) = string(x)

function write_csv(name::AbstractString, df::DataFrame)
    mkpath(RESULTS_DIR)
    path = joinpath(RESULTS_DIR, name)
    open(path, "w") do io
        println(io, join(names(df), ","))
        for r in eachrow(df)
            println(io, join((_fmt(r[c]) for c in names(df)), ","))
        end
    end
    println("  wrote ", relpath(path, pwd()))
    return path
end

# ---- evaluation of one population -----------------------------------------------------
"""
    evaluate_population(cfg, pop; experiment, label=pop.name, ...) -> (runs, trajectories)

Runs every replication of `cfg` for `pop`. `runs` has one row per replication
(run metrics + diagnostics); `trajectories` has block-mean price / CI / welfare series
for plotting. Full histories are not kept (memory).
"""
function evaluate_population(cfg::SimulationConfig, pop::Population; experiment::AbstractString,
                             label::AbstractString = pop.name, window::Integer = min(1000, cfg.T),
                             traj_block::Integer = 100, diagnostics::Bool = true,
                             n_perm::Integer = 200, horizon::Integer = 10)
    p = cfg.params
    runs, traj = NamedTuple[], NamedTuple[]
    for (r, s) in enumerate(replication_seeds(cfg.base_seed, cfg.n_reps))
        h, agents, rng = run_simulation_with_state(cfg, pop, s)
        row = merge(run_metrics(h, p; window = window),
                    (; experiment = String(experiment), population = String(label), rep = r,
                       T = cfg.T, seed = s))
        row = merge(row, NamedTuple{ntuple(i -> Symbol("agent$i"), p.n_firms)}(Tuple(h.agent_labels)))
        if diagnostics
            sy = synchrony_test(h; window = window, n_perm = n_perm, rng = Xoshiro(s))
            dv = deviation_probe(agents, rng, p, h; horizon = horizon)
            row = merge(row, (; sync_p = sy.p_value, sync_null_mean = sy.null_mean,
                                dev_possible = dv.possible, dev_depth = dv.depth,
                                dev_final_gap = dv.final_gap, dev_reversion = dv.reversion))
        end
        push!(runs, row)

        pb  = block_means(h.transaction_price, traj_block)
        wb  = block_means(h.total_welfare, traj_block)
        for b in eachindex(pb)
            push!(traj, (; experiment = String(experiment), population = String(label), rep = r,
                           period_end = b * traj_block, mean_price = pb[b],
                           ci = collusion_index(pb[b], p), mean_welfare = wb[b]))
        end
    end
    return DataFrame(runs), DataFrame(traj)
end

# ---- summaries ------------------------------------------------------------------------------
const SUMMARY_METRICS = [:price_final, :ci_final, :ci_first, :ci_peak_block, :ci_reversion,
                         :cs_final, :ps_final, :welfare_final, :welfare_ratio_final,
                         :deadweight_loss_final, :elevated_share_final, :elevated_share_all,
                         :longest_elevated_run, :persistent, :price_dispersion_final,
                         :all_equal_share_final, :synchrony_final, :sync_p,
                         :dev_depth, :dev_reversion,
                         :profit_firm1, :profit_firm2, :profit_firm3]

"Mean, sd and bootstrap 95% CI across replications, for every metric and population."
function summarize_runs(runs::DataFrame; metrics = SUMMARY_METRICS)
    rows = NamedTuple[]
    for g in groupby(runs, :population)
        for m in metrics
            hasproperty(g, m) || continue
            d = describe_sample(Float64.(g[!, m]); rng = Xoshiro(1))
            push!(rows, (; population = first(g.population), metric = String(m), d...))
        end
    end
    return DataFrame(rows)
end

function print_summary(summary::DataFrame, metrics)
    for pop in unique(summary.population)
        println("\n", pop)
        for m in metrics
            r = summary[(summary.population .== pop) .& (summary.metric .== String(m)), :]
            nrow(r) == 1 || continue
            println("  ", rpad(m, 24), rpad(round(r.mean[1]; digits = 3), 10),
                    "95% CI [", round(r.ci_lo[1]; digits = 3), ", ", round(r.ci_hi[1]; digits = 3),
                    "]   sd ", round(r.sd[1]; digits = 3))
        end
    end
end

const HEADLINE_METRICS = [:price_final, :ci_final, :elevated_share_final, :persistent,
                          :ci_reversion, :welfare_ratio_final, :ps_final, :cs_final,
                          :all_equal_share_final, :dev_depth, :dev_reversion]

# ---- Experiments 1-3 (also reused as controls for Experiment 4) -----------------------------
function run_homogeneous_experiments(cfg::SimulationConfig)
    specs = [("exp1_homog_q",    homogeneous_population(:q)),
             ("exp2_homog_evo",  homogeneous_population(:evo)),
             ("exp3_homog_rule", homogeneous_population(:rule))]
    runs, trajs = DataFrame[], DataFrame[]
    for (name, pop) in specs
        println("  running ", name, " (", pop.name, ")")
        r, t = evaluate_population(cfg, pop; experiment = name)
        push!(runs, r); push!(trajs, t)
    end
    return reduce((a, b) -> vcat(a, b; cols = :union), runs),
           reduce((a, b) -> vcat(a, b; cols = :union), trajs)
end
