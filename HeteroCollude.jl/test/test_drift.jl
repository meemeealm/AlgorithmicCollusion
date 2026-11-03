using Random

@testset "Phase 3c: noise and parameter diversity (rule-based agents)" begin

    @testset "noise off: acts exactly as the rule says and consumes no randomness" begin
        rng = Xoshiro(1)
        ag = AdaptiveRuleAgent(p; noise = 0.0, rng = rng)
        ag.k = 9
        ref = copy(rng)
        @test all(act!(ag, rng, t) == 9 for t in 1:50)
        @test rand(rng) == rand(ref)
    end

    @testset "noise on: always perturbs, within the stated width" begin
        rng = Xoshiro(2)
        ag = AdaptiveRuleAgent(p; noise = 1.0, noise_width = 2, rng = rng)
        ag.k = 9
        acts = [act!(ag, rng, t) for t in 1:500]
        @test all(a -> a != 9 && 7 <= a <= 11, acts)
        @test Set(acts) == Set([7, 8, 10, 11])
    end

    @testset "noise frequency matches the parameter" begin
        rng = Xoshiro(3)
        ag = AdaptiveRuleAgent(p; noise = 0.3, noise_width = 2, rng = rng)
        ag.k = 9
        acts = [act!(ag, rng, t) for t in 1:5000]
        @test abs(count(!=(9), acts) / 5000 - 0.3) < 0.03
    end

    @testset "the rule reacts to the price actually posted" begin
        rng = Xoshiro(4)
        ag = AdaptiveRuleAgent(p; noise = 1.0, noise_width = 2, step = 1, rng = rng)
        ag.k = 9; ag.dir = +1; ag.prev_profit = 0.0
        k = act!(ag, rng, 1)
        observe!(ag, Observation(k, 100.0, 1.0, 15), 1)
        @test ag.k == k + 1
    end

    @testset "parameter validation" begin
        @test_throws ArgumentError AdaptiveRuleAgent(p; noise = -0.1)
        @test_throws ArgumentError AdaptiveRuleAgent(p; noise = 1.1)
        @test_throws ArgumentError AdaptiveRuleAgent(p; noise_width = 0)
    end

    @testset "parameter diversity" begin
        R = RuleParamRanges()
        rng = Xoshiro(5)
        ags = [sample_rule_agent(p, rng) for _ in 1:200]
        @test all(a -> a.step in R.step && a.cut_step in R.cut_step &&
                       R.noise[1] <= a.noise <= R.noise[2] &&
                       a.noise_width in R.noise_width, ags)
        @test Set(a.step for a in ags) == Set(R.step)
        @test Set(a.cut_step for a in ags) == Set(R.cut_step)
        @test Set(a.noise_width for a in ags) == Set(R.noise_width)
        @test length(unique(a.noise for a in ags)) > 100
        @test length(diverse_rule_agents(p, 3, rng)) == 3

        sig(v) = [(x.step, x.cut_step, x.noise, x.noise_width, x.k, x.dir) for x in v]
        @test sig(diverse_rule_agents(p, 3, Xoshiro(6))) == sig(diverse_rule_agents(p, 3, Xoshiro(6)))

        fixed = RuleParamRanges(step = 1:1, cut_step = 2:2, noise = (0.05, 0.05), noise_width = 2:2)
        @test all(a -> (a.step, a.cut_step, a.noise, a.noise_width) == (1, 2, 0.05, 2),
                  diverse_rule_agents(p, 5, Xoshiro(7); ranges = fixed))
        @test_throws ArgumentError sample_rule_agent(p, Xoshiro(1);
                                       ranges = RuleParamRanges(noise = (0.2, 0.1)))
    end

    @testset "identical noise-free agents move in lockstep; diversity plus noise breaks it" begin
        rng = Xoshiro(8)
        same = [AdaptiveRuleAgent(p; noise = 0.0, rng = rng) for _ in 1:3]
        foreach(a -> (a.k = 5; a.dir = +1; a.prev_profit = -Inf), same)
        pr = play(p, same, rng, 300; reset = false)
        @test pr[:, 1] == pr[:, 2] == pr[:, 3]          # perfectly synchronised without any coordination

        mixed = diverse_rule_agents(p, 3, rng)
        foreach(a -> (a.k = 5; a.dir = +1; a.prev_profit = -Inf), mixed)
        pr = play(p, mixed, rng, 2000; reset = false)
        @test !(pr[:, 1] == pr[:, 2] == pr[:, 3])
    end
end
