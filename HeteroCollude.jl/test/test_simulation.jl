using Random

@testset "Phase 4: simulation engine" begin
    cfg = SimulationConfig(T = 600, n_reps = 3, base_seed = 11)

    @testset "configuration and seeds" begin
        d = SimulationConfig()
        @test d.T == 10_000 && d.n_reps == 30 && d.params.n_firms == 3
        @test_throws ArgumentError SimulationConfig(T = 1)
        @test_throws ArgumentError SimulationConfig(n_reps = 0)
        @test replication_seeds(11, 3) == replication_seeds(11, 3)
        @test replication_seeds(11, 3) == replication_seeds(11, 5)[1:3]   # extendable
        @test replication_seeds(11, 3) != replication_seeds(12, 3)
        @test length(unique(replication_seeds(11, 30))) == 30
    end

    @testset "population specifications" begin
        @test homogeneous_population(:q).kinds == [:q, :q, :q]
        @test mixed_population().kinds == [:q, :evo, :rule]
        @test mixed_population().name == "1Q+1E+1R"
        @test homogeneous_population(:rule).name == "3R"
        @test population_from_counts(2, 0, 1).kinds == [:q, :q, :rule]
        @test_throws ArgumentError Population([:q, :gpt, :evo])
        @test_throws ArgumentError Population([:q])
        comps = population_compositions(3)
        @test length(comps) == 10
        @test all(c -> sum(c) == 3, comps)
        @test (1, 1, 1) in comps && (3, 0, 0) in comps && (0, 0, 3) in comps
        @test length(unique(comps)) == 10
    end

    @testset "agent construction" begin
        rng = Xoshiro(1)
        a = build_agent(:q, homogeneous_population(:q), p, 500, rng)
        @test a.schedule isa LinearDecay && a.schedule.T == 500       # tied to run length
        custom = Population([:q, :q, :q]; q_kwargs = (schedule = ConstantEpsilon(0.1),))
        @test build_agent(:q, custom, p, 500, rng).schedule isa ConstantEpsilon
        fx = homogeneous_population(:rule; rule_ranges = fixed_rule_ranges())
        ags = [build_agent(:rule, fx, p, 100, rng) for _ in 1:3]
        @test all(x -> (x.step, x.cut_step, x.noise, x.noise_width) == (1, 2, 0.0, 2), ags)
        @test_throws ArgumentError build_agent(:gpt, fx, p, 100, rng)
    end

    @testset "history is a faithful record" begin
        h = run_simulation(cfg, mixed_population(), 123)
        @test size(h.price_indices) == (600, 3) && size(h.profits) == (600, 3)
        @test length(h.transaction_price) == 600
        @test h.population == "1Q+1E+1R" && h.seed == 123
        @test h.posted_prices == p.prices[h.price_indices]
        @test all(1 .<= h.price_indices .<= n_actions(p))
        @test all(1:600) do t
            o = clear_market(p, h.price_indices[t, :])
            h.profits[t, :] == o.profits &&
            h.transaction_price[t] == o.transaction_price &&
            h.quantity[t] == o.quantity &&
            h.total_welfare[t] ≈ h.consumer_surplus[t] + h.producer_surplus[t] &&
            h.producer_surplus[t] ≈ sum(h.profits[t, :])
        end
        @test h.agent_labels == ["Q-learning", "Evolutionary", "Rule-based"]
        @test h.agent_params[1][:alpha] == 0.1 && h.agent_params[1][:gamma] == 0.95
        @test h.agent_params[2][:generations] == 3.0                  # 600 ÷ (10 × 20)
        R = RuleParamRanges()
        rp = h.agent_params[3]
        @test rp[:step] in R.step && rp[:cut_step] in R.cut_step && rp[:noise_width] in R.noise_width
        @test R.noise[1] <= rp[:noise] <= R.noise[2]
    end

    @testset "reproducibility" begin
        pop = mixed_population()
        h1, h2 = run_simulation(cfg, pop, 5), run_simulation(cfg, pop, 5)
        @test h1.price_indices == h2.price_indices && h1.profits == h2.profits
        @test h1.agent_params == h2.agent_params
        h3 = run_simulation(cfg, pop, 6)
        @test h1.price_indices != h3.price_indices
    end

    @testset "replications" begin
        pop = mixed_population()
        hs = run_replications(cfg, pop)
        seeds = replication_seeds(cfg.base_seed, cfg.n_reps)
        @test length(hs) == 3 && [h.seed for h in hs] == seeds
        @test hs[1].price_indices == run_simulation(cfg, pop, seeds[1]).price_indices
        @test hs[1].price_indices != hs[2].price_indices
    end

    @testset "market size mismatch" begin
        @test_throws ArgumentError run_simulation(cfg, Population([:q, :evo]), 1)
    end

    @testset "experiments 1-4 run end to end" begin
        short = SimulationConfig(T = 400, n_reps = 1)
        for pop in (homogeneous_population(:q), homogeneous_population(:evo),
                    homogeneous_population(:rule), mixed_population())
            h = run_simulation(short, pop, 3)
            @test all(isfinite, h.profits) && all(h.profits .>= 0)
            @test all(h.total_welfare .<= benchmarks(p).welfare_c + 1e-9)
        end
    end
end
