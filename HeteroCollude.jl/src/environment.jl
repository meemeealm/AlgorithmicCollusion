# environment.jl
# Economic primitives: parameters, demand, consumer surplus, theoretical benchmarks.
#
# Linear demand:   D(p) = max(0, (a - p) / b)
# Constant marginal cost c, no fixed costs, no capacity constraints.

"""
    MarketParams(; n_firms=3, a=100, b=1, c=10, p_min=10, p_max=90, p_step=5)

Immutable description of the repeated-Bertrand stage game.

Agents choose an *action index* k in 1:n_actions(params); the posted price is
`params.prices[k]`. Working with integer indices (not floats) makes tie detection
exact.

Phase-1 restrictions (so that a transaction price always exists and profits are
never negative by construction):
  * c <= p_min   (no below-cost pricing)
  * p_max < a    (demand is strictly positive at every admissible price)
  * (p_max - p_min) / p_step must be an integer
"""
struct MarketParams
    n_firms::Int
    a::Float64
    b::Float64
    c::Float64
    p_min::Float64
    p_max::Float64
    p_step::Float64
    prices::Vector{Float64}

    function MarketParams(n_firms, a, b, c, p_min, p_max, p_step)
        n_firms >= 2     || throw(ArgumentError("n_firms must be >= 2"))
        b > 0            || throw(ArgumentError("demand slope b must be > 0"))
        p_step > 0       || throw(ArgumentError("p_step must be > 0"))
        c >= 0           || throw(ArgumentError("marginal cost must be >= 0"))
        p_min >= c       || throw(ArgumentError("p_min must be >= marginal cost"))
        p_max >= p_min   || throw(ArgumentError("p_max must be >= p_min"))
        p_max < a        || throw(ArgumentError("p_max must be < demand intercept a"))
        steps = (p_max - p_min) / p_step
        abs(steps - round(steps)) < 1e-9 ||
            throw(ArgumentError("(p_max - p_min) must be a multiple of p_step"))
        prices = [p_min + k * p_step for k in 0:Int(round(steps))]
        return new(n_firms, a, b, c, p_min, p_max, p_step, prices)
    end
end

MarketParams(; n_firms::Integer = 3, a::Real = 100, b::Real = 1, c::Real = 10,
               p_min::Real = 10, p_max::Real = 90, p_step::Real = 5) =
    MarketParams(Int(n_firms), Float64(a), Float64(b), Float64(c),
                 Float64(p_min), Float64(p_max), Float64(p_step))

"Number of discrete price actions."
n_actions(p::MarketParams) = length(p.prices)

"Posted price associated with action index `k`."
price_of(p::MarketParams, k::Integer) = p.prices[k]

"Market demand D(price) = max(0, (a - price) / b)."
demand(p::MarketParams, price::Real) = max(0.0, (p.a - price) / p.b)

"""
    consumer_surplus(p, P)

Area under the demand curve above price `P`:  ∫_P^a (a-x)/b dx = (a-P)^2 / (2b).
With a=100, b=1 this is 0.5 (100-P)^2. Valid because all demanded units are
actually supplied at P (no rationing). Returns 0 for P >= a.
"""
consumer_surplus(p::MarketParams, P::Real) = P >= p.a ? 0.0 : (p.a - P)^2 / (2p.b)

# ---------------------------------------------------------------------------
# Theoretical benchmarks (continuous-price theory)
# ---------------------------------------------------------------------------

"""
Bertrand competitive price. With identical constant marginal cost and
homogeneous goods, any price above c can be undercut profitably, so the
(continuous-price) equilibrium is P_c = c.
"""
competitive_price(p::MarketParams) = p.c

"""
Monopoly price. Maximise (P-c)(a-P)/b: FOC (a-P) - (P-c) = 0  =>  P_m = (a+c)/2.
(The slope b cancels.) With a=100, c=10: P_m = 55.
"""
monopoly_price(p::MarketParams)    = (p.a + p.c) / 2
monopoly_quantity(p::MarketParams) = demand(p, monopoly_price(p))                      # (a-c)/(2b)
monopoly_profit(p::MarketParams)   = (monopoly_price(p) - p.c) * monopoly_quantity(p)  # (a-c)^2/(4b)

"True if `price` coincides with a point of the discrete price grid."
on_grid(p::MarketParams, price::Real) = any(g -> isapprox(price, g; atol = 1e-9), p.prices)

"Reference values for welfare analysis at the competitive and monopoly outcomes."
struct Benchmarks
    Pc::Float64;  Pm::Float64
    Qc::Float64;  Qm::Float64
    profit_c::Float64;  profit_m::Float64      # industry profit
    cs_c::Float64;      cs_m::Float64
    welfare_c::Float64; welfare_m::Float64
    deadweight_loss::Float64                   # welfare_c - welfare_m
end

function benchmarks(p::MarketParams)
    Pc, Pm = competitive_price(p), monopoly_price(p)
    Qc, Qm = demand(p, Pc), demand(p, Pm)
    profit_c = (Pc - p.c) * Qc            # = 0
    profit_m = (Pm - p.c) * Qm
    cs_c, cs_m = consumer_surplus(p, Pc), consumer_surplus(p, Pm)
    wc, wm = cs_c + profit_c, cs_m + profit_m
    return Benchmarks(Pc, Pm, Qc, Qm, profit_c, profit_m, cs_c, cs_m, wc, wm, wc - wm)
end

# ---------------------------------------------------------------------------
# Experiment-wide defaults (single source of truth)
# ---------------------------------------------------------------------------
"Default number of periods per simulation run."
const DEFAULT_PERIODS = 10_000
"Default number of independent replications per configuration."
const DEFAULT_REPLICATIONS = 30
