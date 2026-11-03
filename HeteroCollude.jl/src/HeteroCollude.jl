module HeteroCollude

using Random
using Statistics

include("environment.jl")
include("market.jl")
include("agents.jl")
include("qlearning.jl")
include("evolutionary.jl")
include("rulebased.jl")
include("simulation.jl")
include("metrics.jl")
include("diagnostics.jl")

export MarketParams, n_actions, price_of, demand, consumer_surplus,
       competitive_price, monopoly_price, monopoly_quantity, monopoly_profit,
       Benchmarks, benchmarks, on_grid, DEFAULT_PERIODS, DEFAULT_REPLICATIONS,
       MarketOutcome, clear_market, best_deviation_gain, is_nash,
       AbstractAgent, Observation, make_observation, act!, observe!, reset!,
       FixedPriceAgent, argmax_random_ties,
       EpsilonSchedule, ConstantEpsilon, LinearDecay, ExponentialDecay, epsilon,
       QLearningAgent, state_index, greedy_action, random_play_q0,
       EvolutionaryAgent, relative_state, evolve!,
       AdaptiveRuleAgent, RuleParamRanges, sample_rule_agent, diverse_rule_agents,
       Population, homogeneous_population, mixed_population, population_from_counts,
       population_compositions, fixed_rule_ranges, build_agent, agent_label, agent_params,
       SimulationConfig, replication_seeds, SimulationHistory,
       run_simulation, run_simulation_with_state, run_replications,
       collusion_index, block_means, longest_run, highest_symmetric_nash_price,
       describe_sample, permutation_test, synchrony, synchrony_test, run_metrics,
       force_action!, deviation_probe

end # module
