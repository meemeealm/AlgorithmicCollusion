using Test
using HeteroCollude

const p = MarketParams()                       # default parameters from the project spec
ix(price) = Int(round((price - 10) / 5)) + 1   # price -> action index on the default grid

@testset "Phase 1: market environment" begin

    @testset "parameters and grid" begin
        @test p.n_firms == 3
        @test n_actions(p) == 17
        @test p.prices[1] == 10 && p.prices[end] == 90
        @test all(diff(p.prices) .== 5)
        @test price_of(p, ix(55)) == 55
        @test_throws ArgumentError MarketParams(n_firms = 1)
        @test_throws ArgumentError MarketParams(p_max = 100)     # p_max must be < a
        @test_throws ArgumentError MarketParams(p_min = 5)       # below cost
        @test_throws ArgumentError MarketParams(p_step = 7)      # grid mismatch
    end

    @testset "demand and surplus formulas" begin
        @test demand(p, 10) == 90
        @test demand(p, 55) == 45
        @test demand(p, 100) == 0
        @test demand(p, 120) == 0            # no negative demand
        @test consumer_surplus(p, 10) == 4050
        @test consumer_surplus(p, 55) == 1012.5
        @test consumer_surplus(p, 100) == 0
    end

    @testset "theoretical benchmarks" begin
        B = benchmarks(p)
        @test B.Pc == 10
        @test B.Pm == 55
        @test on_grid(p, B.Pc) && on_grid(p, B.Pm)
        @test B.Qc == 90 && B.Qm == 45
        @test B.profit_c == 0
        @test B.profit_m == 2025
        @test B.welfare_c == 4050
        @test B.welfare_m == 3037.5
        @test B.deadweight_loss == 1012.5
        mono_profit(price) = (price - p.c) * demand(p, price)
        @test p.prices[argmax(mono_profit.(p.prices))] == 55
    end

    @testset "market clearing" begin
        o = clear_market(p, fill(ix(10), 3))              # competitive outcome
        @test o.transaction_price == 10
        @test o.quantity == 90
        @test all(o.profits .== 0)
        @test o.total_welfare == 4050

        o = clear_market(p, fill(ix(55), 3))              # tacit monopoly
        @test o.transaction_price == 55
        @test o.quantities == fill(15.0, 3)
        @test o.profits == fill(675.0, 3)
        @test o.producer_surplus == 2025
        @test o.consumer_surplus == 1012.5
        @test o.total_welfare == 3037.5

        o = clear_market(p, [ix(55), ix(55), ix(50)])     # undercutter takes the market
        @test o.transaction_price == 50
        @test o.profits == [0.0, 0.0, 2000.0]
        @test o.posted_prices == [55.0, 55.0, 50.0]

        o = clear_market(p, [ix(50), ix(50), ix(60)])     # tie at the lowest price
        @test o.quantities == [25.0, 25.0, 0.0]
        @test o.profits == [1000.0, 1000.0, 0.0]

        @test_throws ArgumentError clear_market(p, [1, 2])
        @test_throws ArgumentError clear_market(p, [1, 2, 18])
    end

    @testset "accounting identities (all 17^3 price profiles)" begin
        K = n_actions(p)
        Wc = benchmarks(p).welfare_c
        for prof in Iterators.product(1:K, 1:K, 1:K)
            o = clear_market(p, collect(prof))
            @test sum(o.quantities) ≈ o.quantity
            @test o.producer_surplus ≈ (o.transaction_price - p.c) * o.quantity
            @test o.total_welfare ≈ o.consumer_surplus + o.producer_surplus
            @test o.total_welfare <= Wc + 1e-9      # competitive welfare is the ceiling
            @test all(o.profits .>= 0)
        end
    end

    @testset "symmetry under firm relabelling" begin
        o1 = clear_market(p, [ix(30), ix(45), ix(30)])
        o2 = clear_market(p, [ix(45), ix(30), ix(30)])
        @test sort(o1.profits) == sort(o2.profits)
        @test o1.transaction_price == o2.transaction_price
    end

    @testset "industry-profit maximum is the monopoly outcome" begin
        K = n_actions(p)
        best = -Inf
        arg = Float64[]
        for prof in Iterators.product(1:K, 1:K, 1:K)
            o = clear_market(p, collect(prof))
            if o.producer_surplus > best + 1e-9
                best = o.producer_surplus
                arg = [o.transaction_price]
            elseif abs(o.producer_surplus - best) <= 1e-9
                push!(arg, o.transaction_price)
            end
        end
        @test best ≈ benchmarks(p).profit_m
        @test unique(arg) == [55.0]
    end

    @testset "stage-game Nash equilibria on the discrete grid" begin
        @test is_nash(p, fill(ix(10), 3))        # competitive benchmark
        @test is_nash(p, fill(ix(15), 3))        # near-competitive equilibrium (see notes)
        @test !is_nash(p, fill(ix(55), 3))       # monopoly outcome is not an equilibrium
        @test !is_nash(p, fill(ix(90), 3))
        @test best_deviation_gain(p, fill(ix(55), 3), 1) ≈ 2000 - 675
        sym_ne = [k for k in 1:n_actions(p) if is_nash(p, fill(k, 3))]
        @test sym_ne == [ix(10), ix(15)]
        base = clear_market(p, fill(ix(15), 3)).profits[1]    # all-15 is a STRICT equilibrium
        for k in 1:n_actions(p)
            k == ix(15) && continue
            @test clear_market(p, [k, ix(15), ix(15)]).profits[1] < base
        end
    end

    @testset "collusion index" begin
        @test collusion_index(10, p) == 0
        @test collusion_index(55, p) == 1
        @test collusion_index(15, p) ≈ 5 / 45
        @test collusion_index(90, p) ≈ 80 / 45           # > 1, not clamped
        @test collusion_index(5, p) < 0                  # < 0, not clamped
        @test collusion_index([10, 55], p) == [0.0, 1.0]
    end
end

include("helpers.jl")
include("test_qlearning.jl")
include("test_evolutionary.jl")
include("test_rulebased.jl")
include("test_drift.jl")
include("test_simulation.jl")
include("test_metrics.jl")
