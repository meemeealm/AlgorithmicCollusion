using Random, Statistics

# A rival that posts the lowest price it saw last period (price matching).
mutable struct MatchAgent <: AbstractAgent
    next::Int
    last_action::Int
end
HeteroCollude.act!(a::MatchAgent, rng::AbstractRNG, t::Integer) = (a.last_action = a.next; a.next)
HeteroCollude.observe!(a::MatchAgent, obs::Observation, t::Integer) =
    (a.next = min(obs.own_action, obs.min_rival_action); nothing)
HeteroCollude.reset!(a::MatchAgent, rng::AbstractRNG) = a

# Build a history from a T × n matrix of price indices.
function synthetic_history(idx::Matrix{Int})
    T, n = size(idx)
    outs = [clear_market(p, idx[t, :]) for t in 1:T]
    rows(f) = permutedims(reduce(hcat, [f(o) for o in outs]))
    return SimulationHistory("synthetic", UInt64(1), T, n, fill("x", n),
        [Dict{Symbol,Float64}() for _ in 1:n], idx,
        rows(o -> o.posted_prices), [o.transaction_price for o in outs], [o.quantity for o in outs],
        rows(o -> o.quantities), rows(o -> o.profits),
        [o.consumer_surplus for o in outs], [o.producer_surplus for o in outs],
        [o.total_welfare for o in outs])
end
const_hist(prices, T) = synthetic_history(repeat(reshape([ix(x) for x in prices], 1, :), T, 1))

function stepseries(rng, T)               # piecewise-constant random price indices
    out = Int[]
    while length(out) < T
        append!(out, fill(rand(rng, 1:17), rand(rng, 1:6)))
    end
    return out[1:T]
end

@testset "Phase 5: metrics and diagnostics" begin

    @testset "helpers" begin
        @test block_means([1, 2, 3, 4, 5, 6, 7], 3) == [2.0, 5.0]
        @test longest_run(Bool[1, 1, 0, 1, 1, 1, 0]) == 3
        @test longest_run(falses(4)) == 0
        @test highest_symmetric_nash_price(p) == 15
    end

    @testset "tacit-monopoly history" begin
        m = run_metrics(const_hist([55, 55, 55], 2000), p; window = 1000)
        @test m.price_final == 55 && m.ci_final == 1.0 && m.ci_first == 1.0
        @test m.ps_final == 2025 && m.cs_final == 1012.5 && m.welfare_final == 3037.5
        @test m.welfare_ratio_final ≈ 0.75
        @test m.deadweight_loss_final == 1012.5
        @test m.profit_firm1 == 675 && m.profit_firm3 == 675
        @test m.threshold == 15
        @test m.elevated_share_final == 1 && m.longest_elevated_run == 2000 && m.persistent
        @test m.ci_reversion == 0
        @test m.price_dispersion_final == 0 && m.all_equal_share_final == 1
        @test isnan(m.synchrony_final)
    end

    @testset "competitive history" begin
        m = run_metrics(const_hist([10, 10, 10], 1000), p)
        @test m.ci_final == 0 && m.welfare_ratio_final == 1 && m.ps_final == 0
        @test m.elevated_share_final == 0 && m.longest_elevated_run == 0 && !m.persistent
    end

    @testset "near-competitive Nash price is not 'elevated'" begin
        m = run_metrics(const_hist([15, 15, 15], 1000), p)
        @test m.elevated_share_final == 0 && !m.persistent
        @test m.ci_final ≈ 1 / 9
    end

    @testset "transient episode is not persistent" begin
        idx = vcat(repeat([ix(55) ix(55) ix(55)], 1500, 1), repeat([ix(10) ix(10) ix(10)], 1500, 1))
        m = run_metrics(synthetic_history(idx), p; window = 1000)
        @test m.ci_first == 1 && m.ci_final == 0
        @test m.ci_peak_block == 1 && m.ci_reversion == 1
        @test !m.persistent
        @test m.longest_elevated_run == 1500 && m.elevated_share_all ≈ 0.5
    end

    @testset "dispersion and unequal prices" begin
        m = run_metrics(const_hist([50, 60, 70], 1000), p)
        @test m.price_final == 50
        @test m.price_dispersion_final ≈ 10
        @test m.all_equal_share_final == 0
    end

    @testset "synchrony and permutation null" begin
        rng = Xoshiro(21)
        s = stepseries(rng, 1000)
        h_same = synthetic_history(hcat(s, s, s))
        @test synchrony(h_same.posted_prices) ≈ 1
        r = synchrony_test(h_same; window = 1000, n_perm = 200, rng = Xoshiro(1))
        @test r.observed ≈ 1 && r.p_value < 0.05

        h_ind = synthetic_history(hcat(stepseries(rng, 1000), stepseries(rng, 1000), stepseries(rng, 1000)))
        r2 = synchrony_test(h_ind; window = 1000, n_perm = 200, rng = Xoshiro(1))
        @test abs(r2.observed) < 0.3 && r2.p_value > 0.001
    end

    @testset "sample statistics" begin
        d = describe_sample([1, 2, 3, 4, 5, NaN]; rng = Xoshiro(1))
        @test d.n == 5 && d.mean == 3 && d.min == 1 && d.max == 5 && d.median == 3
        @test d.sd ≈ sqrt(2.5) && d.ci_lo < 3 < d.ci_hi
        @test describe_sample(Float64[]).n == 0 && isnan(describe_sample(Float64[]).mean)

        x = collect(0.0:0.1:2.9)
        far = permutation_test(x, x .+ 10; B = 2000, rng = Xoshiro(1))
        @test far.diff ≈ -10 && far.p_value < 0.01
        same = permutation_test(x, x; B = 500, rng = Xoshiro(1))
        @test same.diff == 0 && same.p_value == 1.0
    end

    @testset "deviation probe" begin
        hist = const_hist([55, 55, 55], 5)
        agents = AbstractAgent[FixedPriceAgent(ix(55)), MatchAgent(ix(55), 0), MatchAgent(ix(55), 0)]
        r = deviation_probe(agents, Xoshiro(1), p, hist; horizon = 10)
        # rivals match the undercut from period 2 on and never recover
        @test r.possible
        @test r.gap == vcat(0.0, fill(-5.0, 9))
        @test r.gap_early == -5 && r.gap_late == -5 && r.reversion == 0
        # no randomness in these agents, so the placebo is exactly zero
        @test all(r.placebo_gap .== 0) && r.plc_gap_early == 0 && r.plc_reversion == 0

        floor_hist = const_hist([10, 10, 10], 5)
        r0 = deviation_probe(agents, Xoshiro(1), p, floor_hist; horizon = 6)
        @test !r0.possible && length(r0.gap) == 6 && all(isnan, r0.gap)
        @test isnan(r0.gap_early) && isnan(r0.plc_reversion)
        @test_throws ArgumentError deviation_probe(agents, Xoshiro(1), p, hist; horizon = 4)
    end

    @testset "metrics and probe on a real run" begin
        cfg = SimulationConfig(T = 400, n_reps = 1)
        h, agents, rng = run_simulation_with_state(cfg, mixed_population(), 3)
        @test h.price_indices == run_simulation(cfg, mixed_population(), 3).price_indices
        m = run_metrics(h, p; window = 100)
        @test isfinite(m.price_final) && 0 <= m.elevated_share_final <= 1
        @test m.welfare_ratio_final <= 1 + 1e-9 && haskey(m, :profit_firm3)
        dv = deviation_probe(agents, rng, p, h; horizon = 5)
        @test length(dv.gap) == 5 && length(dv.placebo_gap) == 5
        if dv.possible
            @test all(isfinite, dv.gap) && all(isfinite, dv.placebo_gap)
            @test isfinite(dv.reversion) && isfinite(dv.plc_reversion)
        end
        @test_throws ArgumentError run_metrics(h, p; window = 401)
    end
end