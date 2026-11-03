# Phase 6: figures for Experiments 1-5, read from the CSVs in results/.
#
# One-time setup:
#   julia --project=. -e 'using Pkg; Pkg.add(["CSV", "Plots", "DataFrames"])'
# Run (after homogeneous.jl, heterogeneous.jl and sensitivity.jl):
#   julia --project=. experiments/visualize.jl
#
# Output: results/figures/*.png and *.svg
# Conventions used in every figure
#   * one colour per population (below), same order on every axis
#   * price axes share limits; reference lines (competitive Pc, stage-Nash threshold,
#     monopoly Pm) are drawn only when they fall inside the axis range
#   * lines/bars are means over replications; bands are 5th-95th percentiles across
#     replications; error bars are bootstrap 95% CIs of the mean

using HeteroCollude, Statistics, Random, DataFrames, CSV, Plots
using Plots.Measures

const RESULTS = normpath(joinpath(@__DIR__, "..", "results"))
const FIGDIR  = joinpath(RESULTS, "figures")
const FORMATS = ["png", "svg"]

default(framestyle = :box, grid = true, gridalpha = 0.2, dpi = 200,
        legendfontsize = 8, guidefontsize = 10, tickfontsize = 8, titlefontsize = 10,
        left_margin = 10mm, bottom_margin = 8mm, top_margin = 5mm, right_margin = 4mm)

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
    df = CSV.read(path, DataFrame; missingstring = "NA")
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
    nrow(r) == 1 || return (mean = NaN, ci_lo = NaN, ci_hi = NaN, n = 0)
    return (mean = r.mean[1], ci_lo = r.ci_lo[1], ci_hi = r.ci_hi[1], n = r.n[1])
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

"Mean ± bootstrap CI as one point with error bars (one at a time so colours can differ)."
function dot!(p, x, m, lo, hi, color; ms = 7)
    (isfinite(m) && isfinite(lo) && isfinite(hi)) || return p
    scatter!(p, [x], [m]; yerror = ([m - lo], [hi - m]), markercolor = color, markersize = ms,
             markerstrokecolor = :black, linecolor = :black, label = "")
    return p
end

# ---- Figures 1-2: trajectories ------------------------------------------------------------------------
function trajectory_grid(traj, col; ylabel, ylims, refs)
    refs = [r for r in refs if ylims[1] <= r[1] <= ylims[2]]      # draw only visible reference lines
    pops = present(traj)
    panels = []
    for pop in pops
        xs, m, lo, hi = band(traj, pop, col)
        p = plot(; title = DISPLAY[pop], xlabel = "Period", ylabel = ylabel, ylims = ylims,
                 legend = :topright, legendfontsize = 7)
        for (v, txt, st) in refs
            hline!(p, [v]; color = :gray, linestyle = st, lw = 1, label = txt)
        end
        plot!(p, xs, m; ribbon = (m .- lo, hi .- m), fillalpha = 0.2, color = COLORS[pop], lw = 2,
              label = "Mean (5–95% band)")
        push!(panels, p)
    end
    return plot(panels...; layout = (ceil(Int, length(pops) / 2), 2), size = (1050, 760))
end

price_refs() = [(Pc, "Competitive Pc = $(round(Int, Pc))", :dash),
                (Pn, "Stage-Nash threshold = $(round(Int, Pn))", :dot),
                (Pm, "Monopoly Pm = $(round(Int, Pm))", :dash)]
ci_refs()    = [(0.0, "CI = 0 (competitive)", :dash),
                (collusion_index(Pn, P0), "CI at stage-Nash threshold", :dot),
                (1.0, "CI = 1 (monopoly)", :dash)]

fig1(traj) = trajectory_grid(traj, :mean_price; ylabel = "Mean transaction price",
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

function fig2(traj)                                   # y-range follows the data (shared by all panels)
    top = maximum(maximum(band(traj, p, :ci)[4]) for p in present(traj))
    return trajectory_grid(traj, :ci; ylabel = "Collusion index",
                           ylims = (-0.05, max(0.2, 1.15 * top)), refs = ci_refs())
end

# ---- Figure 3: welfare comparison ---------------------------------------------------------------------
function fig3(summary)
    pops = present(summary)
    xs = 1:length(pops)
    cs = [stat(summary, p, "cs_final").mean for p in pops]
    ps = [stat(summary, p, "ps_final").mean for p in pops]
    p = bar(xs, cs .+ ps; color = PS_COLOR, label = "Producer surplus", bar_width = 0.6,
            xticks = (xs, [SHORT[q] for q in pops]), ylabel = "Welfare per period (final window)",
            ylims = (0, 4600), legend = :outerright, size = (950, 480))
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

# ---- Figure 4: distribution of final prices across replications -----------------------------------------
function fig4(runs)
    pops = present(runs)
    rng = Xoshiro(1)
    p = plot(; ylabel = "Mean transaction price, final window", ylims = (5, 95),
             xticks = (1:length(pops), [SHORT[q] for q in pops]), legend = :outerright, size = (950, 520))
    for (v, txt, st) in price_refs()
        hline!(p, [v]; color = :gray, linestyle = st, lw = 1, label = txt)
    end
    for (i, pop) in enumerate(pops)
        y = Float64.(runs[runs.population .== pop, :price_final])
        scatter!(p, i .+ 0.3 .* (rand(rng, length(y)) .- 0.5), y; markercolor = COLORS[pop],
                 markeralpha = 0.6, markerstrokewidth = 0, markersize = 5, label = "")
        scatter!(p, [i], [median(y)]; markershape = :hline, markersize = 18, markercolor = :black, label = "")
        scatter!(p, [i], [mean(y)]; markershape = :diamond, markersize = 8, markercolor = :white,
                 markerstrokecolor = :black, label = "")
    end
    scatter!(p, [NaN], [NaN]; markershape = :diamond, markercolor = :white, markerstrokecolor = :black, label = "Mean")
    scatter!(p, [NaN], [NaN]; markershape = :hline, markercolor = :black, label = "Median")
    xlims!(p, 0.4, length(pops) + 0.6)
    return p
end

# ---- Figure 5: headline metrics -------------------------------------------------------------------------------
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
    layout = (2, 2), size = (950, 720))

# ---- Figure 6: coordination diagnostics -------------------------------------------------------------------------
"Forced-undercut statistic (filled) next to its placebo (hollow); n = replications where a probe was possible."
function paired_panel(summary, mdev, mplc, ylabel)
    pops = present(summary)
    labels = [string(SHORT[q], " (n=", stat(summary, q, mdev).n, ")") for q in pops]
    p = plot(; ylabel = ylabel, xticks = (1:length(pops), labels), legend = :best, legendfontsize = 7)
    hline!(p, [0.0]; color = :gray, linestyle = :dash, label = "")
    for (i, pop) in enumerate(pops)
        a, b = stat(summary, pop, mdev), stat(summary, pop, mplc)
        dot!(p, i - 0.15, a.mean, a.ci_lo, a.ci_hi, COLORS[pop]; ms = 8)
        dot!(p, i + 0.15, b.mean, b.ci_lo, b.ci_hi, :white; ms = 8)
    end
    scatter!(p, [NaN], [NaN]; markercolor = :gray, markerstrokecolor = :black, label = "Forced undercut")
    scatter!(p, [NaN], [NaN]; markercolor = :white, markerstrokecolor = :black, label = "Placebo (no undercut)")
    xlims!(p, 0.4, length(pops) + 0.6)
    return p
end

fig6(summary) = plot(
    metric_panel(summary, "synchrony_final", "Co-movement of price changes"; ref = 0.0),
    paired_panel(summary, "dev_gap_early", "plc_gap_early", "Rivals' price gap, periods 2–4"),
    paired_panel(summary, "dev_reversion", "plc_reversion", "Reversion (late minus early gap)");
    layout = (1, 3), size = (1500, 500))

# ---- Figure 7: profits by agent type in the mixed population --------------------------------------------------
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

# ---- Figure 8: sensitivity (Experiment 5) -----------------------------------------------------------------------------
const GROUP_ORDER  = ["exp5a_composition", "exp5b_rule_ablation", "exp5c_q_params",
                      "exp5d_horizon", "exp5e_grid", "exp5f_exploration"]
const GROUPS = Dict("exp5a_composition" => "A. composition", "exp5b_rule_ablation" => "B. rule: noise / diversity",
                    "exp5c_q_params" => "C. Q-learning parameters", "exp5d_horizon" => "D. longer horizon",
                    "exp5e_grid" => "E. finer price grid", "exp5f_exploration" => "F. Q exploration schedule")
# neutral palette, deliberately different from the agent colours used elsewhere
const GROUP_COLORS = Dict(zip(GROUP_ORDER, ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D", "#E5C494"]))

"""
Dot plot of one metric for every Experiment 5 setting. The x-range follows the data;
reference lines are (value, label, style, anchor): anchored lines widen the range,
the others are drawn only if they already fall inside it.
"""
function sensitivity_plot(runs, summary, metric, xlabel; refs = Tuple[])
    pops = unique(runs.population)
    group = Dict(r.population => r.experiment for r in eachrow(unique(runs[:, [:population, :experiment]])))
    stats = [stat(summary, q, metric) for q in pops]
    fin = [s for s in stats if isfinite(s.mean) && isfinite(s.ci_lo) && isfinite(s.ci_hi)]
    lo, hi = minimum(s.ci_lo for s in fin), maximum(s.ci_hi for s in fin)
    for (v, _, _, anchor) in refs
        anchor && ((lo, hi) = (min(lo, v), max(hi, v)))
    end
    pad = 0.06 * (hi - lo)
    xl = (lo - pad, hi + pad)
    n = length(pops)
    ypos = collect(n:-1:1)
    p = plot(; xlabel = xlabel, yticks = (ypos, pops), ytickfontsize = 7, legend = :outerbottom,
             size = (950, 60 + 26 * n + 150), left_margin = 4mm)
    xlims!(p, xl...)
    for (v, txt, st, _) in refs
        xl[1] <= v <= xl[2] && vline!(p, [v]; color = :gray, linestyle = st, lw = 1, label = txt)
    end
    for g in GROUP_ORDER
        scatter!(p, [NaN], [NaN]; markercolor = GROUP_COLORS[g], markerstrokecolor = :black, label = GROUPS[g])
    end
    for (i, pop) in enumerate(pops)
        s = stats[i]
        (isfinite(s.mean) && isfinite(s.ci_lo) && isfinite(s.ci_hi)) || continue
        scatter!(p, [s.mean], [ypos[i]]; xerror = ([s.mean - s.ci_lo], [s.ci_hi - s.mean]),
                 markercolor = GROUP_COLORS[group[pop]], markersize = 7, markerstrokecolor = :black,
                 linecolor = :black, label = "")
    end
    return p
end

fig8a(runs, summary) = sensitivity_plot(runs, summary, "price_final", "Mean transaction price, final window";
    refs = [(Pc, "Competitive Pc", :dash, true), (Pn, "Stage-Nash threshold", :dot, true),
            (Pm, "Monopoly Pm", :dash, false)])
fig8b(runs, summary) = sensitivity_plot(runs, summary, "welfare_ratio_final", "Welfare / competitive welfare";
    refs = [(1.0, "Competitive welfare", :dash, true),
            (BM.welfare_m / BM.welfare_c, "Monopoly welfare", :dash, false)])

# ---- driver ------------------------------------------------------------------------------------------------------------------
function try_fig(name, f)
    try
        save_fig(f(), name)
    catch e
        println("  skipped $name: ", sprint(showerror, e))
    end
end

function main()
    println("Reading results from ", relpath(RESULTS, pwd()))
    if isfile(joinpath(RESULTS, "exp4_summary.csv"))
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