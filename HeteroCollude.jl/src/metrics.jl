# metrics.jl
# Evaluation metrics, persistence and coordination diagnostics, sample statistics.
# All metrics are DESCRIPTIVE. A high collusion index is a price-based proxy, not
# proof of collusion, and values outside [0, 1] are reported as they are.

"""
    collusion_index(P, params)

CI = (P - Pc) / (Pm - Pc). Values are NOT clamped: CI < 0 or CI > 1 is returned as-is.
"""
collusion_index(P::Real, p::MarketParams) =
    (P - competitive_price(p)) / (monopoly_price(p) - competitive_price(p))
collusion_index(P::AbstractArray, p::MarketParams) = collusion_index.(P, Ref(p))

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

"Means of consecutive non-overlapping blocks; a trailing incomplete block is dropped."
block_means(x::AbstractVector, block::Integer) =
    [mean(@view x[((b - 1) * block + 1):(b * block)]) for b in 1:fld(length(x), block)]

"Length of the longest run of consecutive `true` values."
function longest_run(mask)
    best = cur = 0
    for m in mask
        cur = m ? cur + 1 : 0
        best = max(best, cur)
    end
    return best
end

"""
    highest_symmetric_nash_price(params)

Highest price at which "everyone posts this price" is a stage-game Nash equilibrium
on the discrete grid (15 for the default market). Prices at or below it are
consistent with non-collusive play; "elevated" means strictly above it.
"""
highest_symmetric_nash_price(p::MarketParams) =
    maximum(p.prices[k] for k in 1:n_actions(p) if is_nash(p, fill(k, p.n_firms)))

# ---------------------------------------------------------------------------
# Sample statistics across replications
# ---------------------------------------------------------------------------

"""
    describe_sample(x; rng, B=2000, level=0.95)

Mean, sd, bootstrap percentile CI of the mean, median, min, max. Non-finite values
are dropped (and `n` reports how many remain).
"""
function describe_sample(x; rng::AbstractRNG = Xoshiro(0), B::Integer = 2000, level::Real = 0.95)
    v = filter(isfinite, collect(Float64, x))
    n = length(v)
    n == 0 && return (; n = 0, mean = NaN, sd = NaN, ci_lo = NaN, ci_hi = NaN,
                        median = NaN, min = NaN, max = NaN)
    boots = [mean(v[rand(rng, 1:n, n)]) for _ in 1:B]
    α = (1 - level) / 2
    lo, hi = quantile(boots, (α, 1 - α))
    return (; n, mean = mean(v), sd = n > 1 ? std(v) : NaN, ci_lo = lo, ci_hi = hi,
              median = median(v), min = minimum(v), max = maximum(v))
end

"""
    permutation_test(x, y; B=10_000, rng)

Two-sided permutation test for a difference in means (unpaired). Returns the
observed difference `mean(x) - mean(y)` and the p-value.
"""
function permutation_test(x, y; B::Integer = 10_000, rng::AbstractRNG = Xoshiro(0))
    a = filter(isfinite, collect(Float64, x))
    b = filter(isfinite, collect(Float64, y))
    (isempty(a) || isempty(b)) && return (; diff = NaN, p_value = NaN)
    obs = mean(a) - mean(b)
    pooled = vcat(a, b)
    na = length(a)
    hits = 0
    for _ in 1:B
        shuffle!(rng, pooled)
        d = mean(@view pooled[1:na]) - mean(@view pooled[(na + 1):end])
        abs(d) >= abs(obs) - 1e-12 && (hits += 1)
    end
    return (; diff = obs, p_value = (hits + 1) / (B + 1))
end

# ---------------------------------------------------------------------------
# Co-movement diagnostics (price data only)
# ---------------------------------------------------------------------------

"""
    synchrony(X)

Mean pairwise Pearson correlation between firms' period-to-period price CHANGES
(X is periods × firms). Pairs with a constant series are skipped; `NaN` if none remain.
"""
function synchrony(X::AbstractMatrix)
    d = diff(X; dims = 1)
    n = size(d, 2)
    cs = Float64[]
    for i in 1:(n - 1), j in (i + 1):n
        a, b = d[:, i], d[:, j]
        (std(a) > 0 && std(b) > 0) && push!(cs, cor(a, b))
    end
    return isempty(cs) ? NaN : mean(cs)
end

"""
    synchrony_test(history; window, n_perm=200, rng)

Is co-movement stronger than for independent firms with the same individual price
dynamics? The null circularly shifts each firm's series by a random lag, which keeps
each firm's own behaviour and destroys alignment between firms. One-sided p-value.
A small p-value shows firms move together; it does NOT say why (common adaptation
and coordination both produce it).
"""
function synchrony_test(h::SimulationHistory; window::Integer = min(1000, h.T),
                        n_perm::Integer = 200, rng::AbstractRNG = Xoshiro(0))
    X = h.posted_prices[(h.T - window + 1):h.T, :]
    obs = synchrony(X)
    isnan(obs) && return (; observed = NaN, null_mean = NaN, p_value = NaN)
    null = zeros(n_perm)
    for b in 1:n_perm
        Y = similar(X)
        for j in axes(X, 2)
            Y[:, j] = circshift(X[:, j], rand(rng, 1:(window - 1)))
        end
        null[b] = synchrony(Y)
    end
    valid = filter(isfinite, null)
    p = (1 + count(>=(obs - 1e-12), valid)) / (length(valid) + 1)
    return (; observed = obs, null_mean = mean(valid), p_value = p)
end

# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

"""
    run_metrics(history, params; window=min(1000, T), threshold=highest_symmetric_nash_price(params))

Scalar summary of one run (NamedTuple). "final" quantities average the last `window`
periods; "first" the first `window`.

Prices and welfare
  price_final, ci_final, ci_first, price_overall,
  cs_final, ps_final (= industry profit), welfare_final,
  welfare_ratio_final (W / competitive W), deadweight_loss_final (competitive W - W),
  profit_firm1..n (per-firm mean profit in the final window)
Persistence (elevated = transaction price strictly above `threshold`)
  elevated_share_final, elevated_share_all, longest_elevated_run,
  persistent  (every non-overlapping `window`-block in the second half of the run is elevated),
  ci_peak_block, ci_reversion (= ci_peak_block - ci_final; large means a transient episode)
Coordination-looking behaviour
  price_dispersion_final (mean cross-firm sd of posted prices),
  all_equal_share_final (share of periods in which all firms post the same price),
  synchrony_final (see `synchrony`)
"""
function run_metrics(h::SimulationHistory, p::MarketParams; window::Integer = min(1000, h.T),
                     threshold::Real = highest_symmetric_nash_price(p))
    1 <= window <= h.T || throw(ArgumentError("window must be in 1:T"))
    T = h.T
    last_w, first_w = (T - window + 1):T, 1:window
    P = h.transaction_price
    B = benchmarks(p)

    blocks   = block_means(P, window)
    nb       = length(blocks)
    persist  = all(>(threshold), blocks[(fld(nb, 2) + 1):nb])
    elevated = P .> threshold
    ci_final = collusion_index(mean(P[last_w]), p)
    ci_peak  = collusion_index(maximum(blocks), p)

    idx_w    = h.price_indices[last_w, :]
    all_eq   = mean(vec(maximum(idx_w; dims = 2) .== minimum(idx_w; dims = 2)))
    disp     = mean(std(h.posted_prices[last_w, :]; dims = 2))

    W = mean(h.total_welfare[last_w])
    firm_profit = vec(mean(h.profits[last_w, :]; dims = 1))
    per_firm = NamedTuple{ntuple(i -> Symbol("profit_firm$i"), h.n_firms)}(Tuple(firm_profit))

    base = (; window, threshold = Float64(threshold),
              price_final = mean(P[last_w]), ci_final,
              ci_first = collusion_index(mean(P[first_w]), p),
              price_overall = mean(P),
              cs_final = mean(h.consumer_surplus[last_w]),
              ps_final = mean(h.producer_surplus[last_w]),
              welfare_final = W,
              welfare_ratio_final = W / B.welfare_c,
              deadweight_loss_final = B.welfare_c - W,
              elevated_share_final = mean(elevated[last_w]),
              elevated_share_all = mean(elevated),
              longest_elevated_run = longest_run(elevated),
              persistent = persist,
              ci_peak_block = ci_peak,
              ci_reversion = ci_peak - ci_final,
              price_dispersion_final = disp,
              all_equal_share_final = all_eq,
              synchrony_final = synchrony(h.posted_prices[last_w, :]))
    return merge(base, per_firm)
end
