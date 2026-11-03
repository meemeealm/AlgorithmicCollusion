using Random

@testset "Phase 3b: adaptive rule-based agent" begin

    function agent_at(; k, dir, prev)
        ag = AdaptiveRuleAgent(p; step = 1, cut_step = 2, rng = Xoshiro(1))
        ag.last_action = k; ag.dir = dir; ag.prev_profit = prev
        return ag
    end

    @testset "rule branches" begin
        ag = agent_at(k = 10, dir = +1, prev = 100.0)        # lost the market
        observe!(ag, Observation(10, 0.0, 0.0, 5), 1)
        @test ag.k == 8 && ag.dir == -1

        ag = agent_at(k = 5, dir = +1, prev = 100.0)         # captured, improved
        observe!(ag, Observation(5, 200.0, 1.0, 9), 1)
        @test ag.k == 6 && ag.dir == +1

        ag = agent_at(k = 5, dir = +1, prev = 200.0)         # captured, got worse -> reverse
        observe!(ag, Observation(5, 100.0, 1.0, 9), 1)
        @test ag.k == 4 && ag.dir == -1

        ag = agent_at(k = 5, dir = -1, prev = 100.0)         # unchanged profit -> keep direction
        observe!(ag, Observation(5, 100.0, 1.0, 9), 1)
        @test ag.k == 4 && ag.dir == -1

        ag = agent_at(k = 1, dir = -1, prev = 0.0)           # zero profit -> go up
        observe!(ag, Observation(1, 0.0, 1.0, 1), 1)
        @test ag.k == 2 && ag.dir == +1
    end

    @testset "grid boundaries" begin
        K = n_actions(p)
        ag = agent_at(k = K, dir = +1, prev = 1.0)
        observe!(ag, Observation(K, 5.0, 1.0, K), 1)
        @test ag.k == K
        ag = agent_at(k = 2, dir = -1, prev = 0.0)
        observe!(ag, Observation(2, 0.0, 0.0, 1), 1)         # lost market near the bottom
        @test ag.k == 1
    end

    @testset "interface" begin
        rng = Xoshiro(2)
        ag = AdaptiveRuleAgent(p; rng = rng)
        k = act!(ag, rng, 1)
        @test 1 <= k <= n_actions(p)
        @test_throws ArgumentError observe!(ag, Observation(mod1(k + 1, n_actions(p)), 0.0, 0.0, 1), 1)
        @test_throws ArgumentError AdaptiveRuleAgent(p; step = 0)
    end

    @testset "behaviour against fixed rivals" begin
        # Rivals at 50. Hill climbing settles into the cycle 45,40,45,50 (mean profit ≈ 1579).
        rng = Xoshiro(5)
        ag = AdaptiveRuleAgent(p; noise = 0.0, rng = rng)     # noise off: deterministic cycle
        pr = play(p, [ag, FixedPriceAgent(ix(50)), FixedPriceAgent(ix(50))], rng, 3000)
        @test sum(pr[end-999:end, 1]) / 1000 > 1400
    end

    @testset "smoke tests: homogeneous and heterogeneous populations" begin
        rng = Xoshiro(9)
        pops = Dict(
            "rule"   => () -> [AdaptiveRuleAgent(p; rng = rng) for _ in 1:3],
            "evo"    => () -> [EvolutionaryAgent(p; rng = rng) for _ in 1:3],
            "q"      => () -> [QLearningAgent(p; rng = rng) for _ in 1:3],
            "hetero" => () -> AbstractAgent[QLearningAgent(p; rng = rng),
                                            EvolutionaryAgent(p; rng = rng),
                                            AdaptiveRuleAgent(p; rng = rng)])
        for (name, mk) in pops
            pr = play(p, mk(), rng, 2000)
            @test all(isfinite, pr)
            @test all(pr .>= 0)
        end
    end

    @testset "reproducibility" begin
        function run_once(seed)
            rng = Xoshiro(seed)
            return play(p, [AdaptiveRuleAgent(p; rng = rng) for _ in 1:3], rng, 1000)
        end
        @test run_once(7) == run_once(7)
    end
end
