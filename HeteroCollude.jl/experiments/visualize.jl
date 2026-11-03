# Figures for Experiments 1-5, read from the CSVs in results/.
#
# One-time setup:
#   julia --project=. -e 'using Pkg; Pkg.add(["CSV", "Plots", "DataFrames"])'
# Run (after homogeneous.jl, heterogeneous.jl and sensitivity.jl):
#   julia --project=. experiments/visualize.jl
#
# Output: results/figures/*.png and *.svg
# Conventions used in every figure
#   * one colour per population (below), same order on every axis
#   * price axes share limits; reference lines: competitive Pc, stage-Nash threshold, monopoly Pm
#   * lines/bars are means over replications; bands are 5th-95th percentiles across
#     replications; error bars are bootstrap 95% CIs of the mean

using HeteroCollude, Statistics, Random, DataFrames, CSV, Plots

const RESULTS = normpath(joinpath(@__DIR__, "..", "results"))
const FIGDIR  = joinpath(RESULTS, "figures")
const FORMATS = ["svg", "png"] 

default(framestyle = :box, grid = true, gridalpha = 0.2, dpi = 200,
        legendfontsize = 8, guidefontsize = 10, tickfontsize = 8, titlefontsize = 10)

# ---- market benchmarks (from the package, so figures follow the model) ------------------
const P0 = MarketParams()
const Pc, Pm, Pn = competitive_price(P0), monopoly_price(P0), highest_symmetric_nash_price(P0)
const BM = benchmarks(P0)

# ---- consistent naming and colours -------------------------------------------------------------
const ORDER   = ["3Q", "3E", "3R", "1Q+1E+1R"]
const DISPLAY = Dict("3Q" => "Q-learning ×3", "3E" => "Evolutionary ×3",
                     "3R" => "Rule-based ×3", "1Q+1E+1R" => "Mixed (Q + E + R)")
const SHORT   = Dict("3Q" => "Q×3", "3E" => "E×3", "3R" => "R×3", "1Q+1E+1R" => "Mixed")
const COLORS  = Dict("3Q" => "#7F77DD", "3E" => "#1D9E75", "3R" => "#D85A30", "1Q+1E+1R" => "#222222")
const AGENT_COLORS = ["#7F77DD", "#1D9E75", "#D85A30"]            # Q, Evolutionary, Rule-based
const AGENT_NAMES  = ["Q-learning", "Evolutionary", "Rule-based"]
const CS_COLOR, PS_COLOR = "#378ADD", "#EF9F27"

# ---- IO helpers --------------------------------------------------------------------------------------
function readres(name)
    path = joinpath(RESULTS, name)
    isfile(path) || error("missing $(relpath(path, pwd())); run the experiment scripts first")
    df = CSV.read(path, DataFrame; missingstring = "NA") # , types = Dict(:seed => String)
    for c in names(df)
        T = eltype(df[!, c])
        if Missing <: T && nonmissingtype(T) <: Real
            df[!, c] = coalesce.(df[!, c], NaN)
        end
    end
    return df
end

function save_fig(p, name)
    mkpath(FIGDIR)
    for ext in FORMATS
        savefig(p, joinpath(FIGDIR, "$name.$ext"))
    end
    println("  saved results/figures/$name.{", join(FORMATS, ","), "}")
end

present(df) = [p for p in ORDER if p in df.population]

function stat(summary, pop, metric)
    r = summary[(summary.population .== pop) .& (summary.metric .== metric), :]
    nrow(r) == 1 || return (mean = NaN, ci_lo = NaN, ci_hi = NaN)
    return (mean = r.mean[1], ci_lo = r.ci_lo[1], ci_hi = r.ci_hi[1])
end

"Mean and 5th/95th percentile across replications at every block end."
function band(df, pop, col; lo = 0.05, hi = 0.95)
    d = df[df.population .== pop, :]
    xs = sort(unique(d.period_end))
    m, l, h = Float64[], Float64[], Float64[]
    for x in xs
        v = d[d.period_end .== x, col]
        push!(m, mean(v)); push!(l, quantile(v, lo)); push!(h, quantile(v, hi))
    end
    return xs, m, l, h
end

"Mean ± bootstrap CI as points with error bars (one point at a time so colours can differ)."
function dot!(p, x, m, lo, hi, color; ms = 7)
    (isfinite(m) && isfinite(lo) && isfinite(hi)) || return p
    scatter!(p, [x], [m]; yerror = ([m - lo], [hi - m]), markercolor = color, markersize = ms,
             markerstrokecolor = :black, linecolor = :black, label = "")
    return p
end

refline!(p, v, txt, xmax; style = :dash) = (hline!(p, [v]; color = :gray, linestyle = style, lw = 1, label = "");
    annotate!(p, xmax, v, text(txt, 7, :right, :bottom)); p)

# ---- Figure 1: price trajectories ----------------------------------------------------------------
function trajectory_grid(traj, col; ylabel, ylims, refs)
    pops = present(traj)
    panels = []
    for pop in pops
        xs, m, lo, hi = band(traj, pop, col)
        p = plot(; title = DISPLAY[pop], xlabel = "Period", ylabel = ylabel, ylims = ylims, legend = false)
        for (v, txt, st) in refs
            refline!(p, v, txt, maximum(xs); style = st)
        end
        plot!(p, xs, m; ribbon = (m .- lo, hi .- m), fillalpha = 0.2, color = COLORS[pop], lw = 2, label = "")
        push!(panels, p)
    end
    return plot(panels...; layout = (ceil(Int, length(pops) / 2), 2), size = (1000, 700))
end

price_refs() = [(Pc, "Competitive Pc = $(round(Int, Pc))", :dash),
                (Pn, "Stage-Nash threshold = $(round(Int, Pn))", :dot),
                (Pm, "Monopoly Pm = $(round(Int, Pm))", :dash)]
ci_refs()    = [(0.0, "CI = 0 (competitive)", :dash),
                (collusion_index(Pn, P0), "CI at stage-Nash threshold", :dot),
                (1.0, "CI = 1 (monopoly)", :dash)]

fig1(traj)  = trajectory_grid(traj, :mean_price; ylabel = "Mean transaction price (100-period blocks)",
                              ylims = (5, 95), refs = price_refs())

function fig1b(traj)
    p = plot(; xlabel = "Period", ylabel = "Mean transaction price", ylims = (5, 95),
             legend = :outerright, size = (1000, 480))
    for (v, txt, st) in price_refs()
        hline!(p, [v]; color = :gray, linestyle = st, lw = 1, label = txt)
    end
    for pop in present(traj)
        xs, m, _, _ = band(traj, pop, :mean_price)
        plot!(p, xs, m; color = COLORS[pop], lw = 2, label = DISPLAY[pop])
    end
    return p
end

# ---- Figure 2: collusion index trajectories ----------------------------------------------------
fig2(traj) = trajectory_grid(traj, :ci; ylabel = "Collusion index (100-period blocks)",
                             ylims = (-0.1, 1.85), refs = ci_refs())

# ---- Figure 3: welfare comparison ---------------------------------------------------------------------
function fig3(summary)
    pops = present(summary)
    xs = 1:length(pops)
    cs = [stat(summary, p, "cs_final").mean for p in pops]
    ps = [stat(summary, p, "ps_final").mean for p in pops]
    p = bar(xs, cs .+ ps; color = PS_COLOR, label = "Producer surplus", bar_width = 0.6,
            xticks = (xs, [SHORT[q] for q in pops]), ylabel = "Welfare per period (final window)",
            ylims = (0, 4600), legend = :outerright, size = (900, 480))
    bar!(p, xs, cs; color = CS_COLOR, label = "Consumer surplus", bar_width = 0.6)
    for (i, pop) in enumerate(pops)
        w = stat(summary, pop, "welfare_final")
        dot!(p, i, w.mean, w.ci_lo, w.ci_hi, :white; ms = 5)
    end
    scatter!(p, [NaN], [NaN]; markercolor = :white, markerstrokecolor = :black, label = "Total welfare (95% CI)")
    hline!(p, [BM.welfare_c]; color = :gray, linestyle = :dash, label = "Competitive welfare = $(round(BM.welfare_c; digits = 1))")
    hline!(p, [BM.welfare_m]; color = :gray, linestyle = :dot,  label = "Monopoly welfare = $(round(BM.welfare_m; digits = 1))")
    return p
end

# ---- Figure 4: distribution of final prices across replications ----------------------------------------------
function fig4(runs)
    pops = present(runs)
    rng = Xoshiro(1)
    p = plot(; ylabel = "Mean transaction price, final window", ylims = (5, 95),
             xticks = (1:length(pops), [SHORT[q] for q in pops]), legend = false, size = (800, 520))
    for (v, txt, st) in price_refs()
        refline!(p, v, txt, length(pops) + 0.45; style = st)
    end
    for (i, pop) in enumerate(pops)
        y = Float64.(runs[runs.population .== pop, :price_final])
        scatter!(p, i .+ 0.3 .* (rand(rng, length(y)) .- 0.5), y; markercolor = COLORS[pop],
                 markeralpha = 0.6, markerstrokewidth = 0, markersize = 5, label = "")
        scatter!(p, [i], [median(y)]; markershape = :hline, markersize = 18, markercolor = :black, label = "")
        scatter!(p, [i], [mean(y)]; markershape = :diamond, markersize = 8, markercolor = :white,
                 markerstrokecolor = :black, label = "")
    end
    xlims!(p, 0.4, length(pops) + 0.6)
    return p
end

# ---- Figure 5: headline metrics ------------------------------------------------------------------------------------
function metric_panel(summary, metric, ylabel; ylims = nothing, ref = nothing)
    pops = present(summary)
    p = plot(; ylabel = ylabel, xticks = (1:length(pops), [SHORT[q] for q in pops]), legend = false)
    ylims === nothing || ylims!(p, ylims...)
    ref === nothing || hline!(p, [ref]; color = :gray, linestyle = :dash, label = "")
    for (i, pop) in enumerate(pops)
        s = stat(summary, pop, metric)
        dot!(p, i, s.mean, s.ci_lo, s.ci_hi, COLORS[pop]; ms = 9)
    end
    xlims!(p, 0.5, length(pops) + 0.5)
    return p
end

fig5(summary) = plot(
    metric_panel(summary, "elevated_share_final", "Share of periods above Nash threshold"; ylims = (-0.05, 1.05)),
    metric_panel(summary, "persistent", "Share of runs with persistent elevation"; ylims = (-0.05, 1.05)),
    metric_panel(summary, "welfare_ratio_final", "Welfare / competitive welfare"; ylims = (0.6, 1.05), ref = 1.0),
    metric_panel(summary, "all_equal_share_final", "Share of periods with identical prices"; ylims = (-0.05, 1.05));
    layout = (2, 2), size = (900, 700))

# ---- Figure 6: coordination diagnostics ----------------------------------------------------------------------------------
fig6(summary) = plot(
    metric_panel(summary, "synchrony_final", "Co-movement of price changes (mean corr.)"; ylims = (-0.3, 1.05), ref = 0.0),
    metric_panel(summary, "dev_depth", "Rivals' price gap after undercut (price units)"; ref = 0.0),
    metric_panel(summary, "dev_reversion", "Reversion after undercut (price units)"; ref = 0.0);
    layout = (1, 3), size = (1300, 450))

# ---- Figure 7: profits by agent type in the mixed population -------------------------------------------
function fig7(summary)
    pop = "1Q+1E+1R"
    p = plot(; ylabel = "Mean profit per period (final window)", xticks = (1:3, AGENT_NAMES),
             legend = false, size = (700, 480), ylims = (0, monopoly_profit(P0)))
    hline!(p, [monopoly_profit(P0) / P0.n_firms]; color = :gray, linestyle = :dash, label = "")
    annotate!(p, 3.45, monopoly_profit(P0) / P0.n_firms, text("Equal share of monopoly profit", 7, :right, :bottom))
    for i in 1:3
        s = stat(summary, pop, "profit_firm$i")
        bar!(p, [i], [s.mean]; color = AGENT_COLORS[i], bar_width = 0.55, label = "")
        dot!(p, i, s.mean, s.ci_lo, s.ci_hi, :white; ms = 5)
    end
    xlims!(p, 0.4, 3.6)
    return p
end

# ---- Figure 8: sensitivity (Experiment 5) ----------------------------------------------------------------------------------
const GROUPS = Dict("exp5a_composition" => "A. composition", "exp5b_rule_ablation" => "B. rule: noise / diversity",
                    "exp5c_q_params" => "C. Q-learning parameters", "exp5d_horizon" => "D. longer horizon",
                    "exp5e_grid" => "E. finer price grid")
const GROUP_ORDER  = ["exp5a_composition", "exp5b_rule_ablation", "exp5c_q_params", "exp5d_horizon", "exp5e_grid"]
const GROUP_COLORS = Dict(zip(GROUP_ORDER, ["#7F77DD", "#1D9E75", "#D85A30", "#378ADD", "#BA7517"]))

function sensitivity_plot(runs, summary, metric, xlabel; refs = Tuple[], xlims = nothing)
    pops = unique(runs.population)
    group = Dict(r.population => r.experiment for r in eachrow(unique(runs[:, [:population, :experiment]])))
    n = length(pops)
    ypos = collect(n:-1:1)
    p = plot(; xlabel = xlabel, yticks = (ypos, pops), ytickfontsize = 7, legend = :outerbottom,
             size = (900, 60 + 26 * n + 120))
    xlims === nothing || xlims!(p, xlims...)
    for (v, txt, st) in refs
        vline!(p, [v]; color = :gray, linestyle = st, lw = 1, label = txt)
    end
    for g in GROUP_ORDER
        scatter!(p, [NaN], [NaN]; markercolor = GROUP_COLORS[g], markerstrokecolor = :black, label = GROUPS[g])
    end
    for (i, pop) in enumerate(pops)
        s = stat(summary, pop, metric)
        (isfinite(s.mean) && isfinite(s.ci_lo) && isfinite(s.ci_hi)) || continue
        scatter!(p, [s.mean], [ypos[i]]; xerror = ([s.mean - s.ci_lo], [s.ci_hi - s.mean]),
                 markercolor = GROUP_COLORS[group[pop]], markersize = 7, markerstrokecolor = :black,
                 linecolor = :black, label = "")
    end
    return p
end

fig8a(runs, summary) = sensitivity_plot(runs, summary, "price_final", "Mean transaction price, final window";
    refs = [(Pc, "Pc", :dash), (Pn, "Nash threshold", :dot), (Pm, "Pm", :dash)], xlims = (5, 95))
fig8b(runs, summary) = sensitivity_plot(runs, summary, "welfare_ratio_final", "Welfare / competitive welfare";
    refs = [(1.0, "Competitive", :dash), (BM.welfare_m / BM.welfare_c, "Monopoly", :dash)], xlims = (0.6, 1.05))

# ---- driver ---------------------------------------------------------------------------------------------------------------------------
function try_fig(name, f)
    try
        save_fig(f(), name)
    catch e
        println("  skipped $name: ", sprint(showerror, e))
    end
end

function main()
    println("Reading results from ", relpath(RESULTS, pwd()))
    have4 = isfile(joinpath(RESULTS, "exp4_summary.csv"))
    if have4
        traj, runs, summary = readres("exp4_trajectories.csv"), readres("exp4_runs.csv"), readres("exp4_summary.csv")
        try_fig("fig1_price_trajectories",       () -> fig1(traj))
        try_fig("fig1b_price_means_overlay",     () -> fig1b(traj))
        try_fig("fig2_collusion_index",          () -> fig2(traj))
        try_fig("fig3_welfare",                  () -> fig3(summary))
        try_fig("fig4_final_price_distribution", () -> fig4(runs))
        try_fig("fig5_headline_metrics",         () -> fig5(summary))
        try_fig("fig6_coordination_diagnostics", () -> fig6(summary))
        try_fig("fig7_profits_mixed",            () -> fig7(summary))
    else
        println("  exp4_* files not found: run experiments/heterogeneous.jl for figures 1-7")
    end
    if isfile(joinpath(RESULTS, "exp5_summary.csv"))
        runs5, summary5 = readres("exp5_runs.csv"), readres("exp5_summary.csv")
        try_fig("fig8a_sensitivity_price",   () -> fig8a(runs5, summary5))
        try_fig("fig8b_sensitivity_welfare", () -> fig8b(runs5, summary5))
    else
        println("  exp5_* files not found: run experiments/sensitivity.jl for figure 8")
    end
end

main()