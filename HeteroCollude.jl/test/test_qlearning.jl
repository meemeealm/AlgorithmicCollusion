# Phase 2 tests. Included from runtests.jl (uses `p` and `ix` defined there).
using Random

@testset "Phase 2: agent interface and Q-learning" begin

    @testset "observations" begin
        o = clear_market(p, [ix(50), ix(40), ix(45)])
        o1 = make_observation(p, o, 1)
        @test o1.own_action == ix(50) && o1.own_profit == 0 && o1.own_quantity == 0
        @test o1.min_rival_action == ix(40)
        o2 = make_observation(p, o, 2)
        @test o2.own_profit == 30 * 60 && o2.own_quantity == 60
        @test o2.min_rival_action == ix(45)
    end

    @testset "epsilon schedules" begin
        @test epsilon(ConstantEpsilon(0.2), 500) == 0.2
        s = LinearDecay(0.2, 0.01, 1000)
        @test epsilon(s, 1) ≈ 0.2
        @test epsilon(s, 1000) ≈ 0.01
        @test epsilon(s, 5000) ≈ 0.01
        @test 0.01 < epsilon(s, 500) < 0.2
        e = ExponentialDecay(0.2, 0.99, 0.02)
        @test epsilon(e, 1) ≈ 0.2
        @test epsilon(e, 2) ≈ 0.198
        @test epsilon(e, 10_000) == 0.02
        @test_throws ArgumentError LinearDecay(0.2, 0.0, 1)
    end

    @testset "tie-breaking and state encoding" begin
        rng = Xoshiro(1)
        @test argmax_random_ties(rng, [1.0, 5.0, 2.0]) == 2
        seen = Set(argmax_random_ties(rng, zeros(4)) for _ in 1:200)
        @test seen == Set(1:4)
        K = n_actions(p)
        ids = [state_index(K, o, r) for o in 1:K for r in 1:K]
        @test sort(ids) == collect(1:K^2)
    end

    @testset "random-play initial values" begin
        tiny = MarketParams(n_firms = 2, p_min = 10, p_max = 20, p_step = 10)  # prices 10, 20
        q0 = random_play_q0(tiny, 0.95)
        @test q0[1] == 0
        @test q0[2] ≈ 200 / (1 - 0.95)       # 0.5*(0) + 0.5*(10 * 40)
        q0p = random_play_q0(p, 0.95)
        @test length(q0p) == n_actions(p)
        @test q0p[1] == 0 && all(q0p .>= 0)
    end

    @testset "Q-update arithmetic" begin
        rng = Xoshiro(1)
        ag = QLearningAgent(p; q_init = :zero, rng = rng)
        K = ag.K
        s  = state_index(K, 3, 4)
        s′ = state_index(K, 7, 2)
        ag.state = s; ag.last_action = 7
        ag.Q[s′, 5] = 100.0
        observe!(ag, Observation(7, 50.0, 1.0, 2), 1)
        @test ag.Q[s, 7] ≈ 0.1 * (50 + 0.95 * 100)      # = 14.5
        @test ag.Q[s, 1] == 0                            # untouched entries
        @test ag.state == s′
        @test_throws ArgumentError observe!(ag, Observation(8, 0.0, 0.0, 2), 2)
    end

    @testset "defaults follow the 10,000-period design" begin
        @test DEFAULT_PERIODS == 10_000 && DEFAULT_REPLICATIONS == 30
        ag = QLearningAgent(p)
        @test ag.schedule isa LinearDecay && ag.schedule.T == DEFAULT_PERIODS
        @test epsilon(ag.schedule, 1) ≈ 0.2
        @test epsilon(ag.schedule, DEFAULT_PERIODS) ≈ 0.01
        @test ag.K^2 == 289
    end

    @testset "action selection" begin
        rng = Xoshiro(2)
        ag = QLearningAgent(p; schedule = ConstantEpsilon(0.0), q_init = :zero, rng = rng)
        ag.Q[ag.state, 6] = 10.0
        @test all(act!(ag, rng, t) == 6 for t in 1:20)           # ε = 0 -> greedy
        ag2 = QLearningAgent(p; schedule = ConstantEpsilon(1.0), rng = rng)
        acts = [act!(ag2, rng, t) for t in 1:2000]
        @test Set(acts) == Set(1:n_actions(p))                   # ε = 1 -> explores everything
        @test all(1 .<= acts .<= n_actions(p))
    end

    @testset "learning against fixed rivals" begin
        # Rivals post 50 forever. Best response on the grid is 45:
        #   P=45, Q=55, profit = 35*55 = 1925 (vs 666.7 for matching 50).
        rng = Xoshiro(42)
        T = 50_000
        ag = QLearningAgent(p; schedule = LinearDecay(0.2, 0.0, T), rng = rng)
        pr = play(p, [ag, FixedPriceAgent(ix(50)), FixedPriceAgent(ix(50))], rng, T)
        @test sum(pr[end-1999:end, 1]) / 2000 > 1700
    end

    @testset "reproducibility" begin
        function run_once(seed)
            rng = Xoshiro(seed)
            ags = [QLearningAgent(p; rng = rng) for _ in 1:3]
            return play(p, ags, rng, 1000)
        end
        @test run_once(7) == run_once(7)
        @test run_once(7) != run_once(8)
    end
end
