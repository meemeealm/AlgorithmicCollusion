using Random

@testset "Phase 3a: evolutionary agent" begin

    @testset "relative state" begin
        @test relative_state(5, 3) == 1     # undercut
        @test relative_state(5, 5) == 2     # tie
        @test relative_state(5, 9) == 3     # led
    end

    @testset "construction and reset" begin
        K = n_actions(p)
        a1 = EvolutionaryAgent(p; rng = Xoshiro(1))
        @test size(a1.pop) == (4, 10)
        @test all(1 .<= a1.pop .<= K)
        a2 = EvolutionaryAgent(p; rng = Xoshiro(1))
        a3 = EvolutionaryAgent(p; rng = Xoshiro(2))
        @test a1.pop == a2.pop
        @test a1.pop != a3.pop
        @test_throws ArgumentError EvolutionaryAgent(p; n_elite = 10)
        @test_throws ArgumentError EvolutionaryAgent(p; mut_rate = 1.5)
    end

    @testset "policy is followed" begin
        rng = Xoshiro(1)
        ag = EvolutionaryAgent(p; rng = rng)
        ag.pop[:, 1] = [5, 6, 7, 8]
        @test act!(ag, rng, 1) == 8                          # opening gene
        observe!(ag, Observation(8, 0.0, 0.0, 3), 1)         # rival lower -> undercut
        @test act!(ag, rng, 2) == 5
        observe!(ag, Observation(5, 0.0, 0.0, 5), 2)         # tie
        @test act!(ag, rng, 3) == 6
        observe!(ag, Observation(6, 0.0, 0.0, 9), 3)         # led
        @test act!(ag, rng, 4) == 7
        @test_throws ArgumentError observe!(ag, Observation(1, 0.0, 0.0, 9), 4)
    end

    @testset "fitness accounting and generation cycle" begin
        rng = Xoshiro(3)
        ag = EvolutionaryAgent(p; pop_size = 4, n_elite = 2, eval_periods = 3, rng = rng)
        for t in 1:3
            k = act!(ag, rng, t)
            observe!(ag, Observation(k, 12.0, 1.0, k), t)
        end
        @test ag.fitness[1] ≈ 12.0
        @test ag.candidate == 2 && ag.generation == 0
        for t in 4:12
            k = act!(ag, rng, t)
            observe!(ag, Observation(k, 12.0, 1.0, k), t)
        end
        @test ag.generation == 1 && ag.candidate == 1
    end

    @testset "selection, retention, mutation" begin
        K = n_actions(p)
        ag = EvolutionaryAgent(p; pop_size = 6, n_elite = 2, rng = Xoshiro(4))
        ag.pop = repeat(collect(1:6)', 4, 1)                 # column j is all j
        ag.fitness = [1.0, 9.0, 3.0, 8.0, 2.0, 0.0]
        evolve!(ag)
        @test ag.pop[:, 1] == fill(2, 4)                     # best retained
        @test ag.pop[:, 2] == fill(4, 4)                     # second best retained
        @test size(ag.pop) == (4, 6) && all(1 .<= ag.pop .<= K)
        @test all(ag.pop[:, 3:6] .<= 7)                      # children within ±3 of parents 2 or 4
        @test ag.generation == 1

        ag2 = EvolutionaryAgent(p; pop_size = 6, n_elite = 2, mut_rate = 0.0, rng = Xoshiro(5))
        ag2.pop = repeat(collect(1:6)', 4, 1)
        ag2.fitness = [1.0, 9.0, 3.0, 8.0, 2.0, 0.0]
        evolve!(ag2)
        @test all(ag2.pop[1, 3:6] .∈ Ref([2, 4]))            # no mutation: exact copies of elites
    end

    @testset "learning against fixed rivals" begin
        # Rivals post 50 forever; best response on the grid is 45 (profit 1925).
        rng = Xoshiro(11)
        ag = EvolutionaryAgent(p; rng = rng)
        pr = play(p, [ag, FixedPriceAgent(ix(50)), FixedPriceAgent(ix(50))], rng, DEFAULT_PERIODS)
        @test sum(pr[end-1999:end, 1]) / 2000 > 1400
        @test ag.generation == DEFAULT_PERIODS ÷ (ag.pop_size * ag.eval_periods)
    end

    @testset "reproducibility" begin
        function run_once(seed)
            rng = Xoshiro(seed)
            return play(p, [EvolutionaryAgent(p; rng = rng) for _ in 1:3], rng, 2000)
        end
        @test run_once(7) == run_once(7)
        @test run_once(7) != run_once(8)
    end
end
