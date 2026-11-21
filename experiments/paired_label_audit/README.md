# Which of two frozen forecasts has lower Brier loss?

When both binary forecasts are available on a finite pool, their absolute difference gives an exact way to spend a small label budget. Sampling rows in proportion to that difference lowers the variance of a particular unbiased risk-difference estimate by a known amount. Alternatively, revealing distinct rows in decreasing order of the difference narrows an exact worst-case interval as quickly as possible. These guarantees answer different questions. Neither needs labels to choose its first row. [Prior registration](protocol.md) · [Executable access checker](design.py)

## A fixed number of random draws

Let f and g be two frozen positive-class probabilities on each of N rows. Define d = f−g, A = mean |d|, and B = mean[d(f+g−1)]. Both A and B use only forecasts. With binary labels y, put h = d(1−2y), H = mean h. The empirical difference in Brier scores is exactly

    Δ = mean[(f−y)² − (g−y)²] = B + H.

Draw an index I with replacement, with probability qᵢ = |dᵢ|/(NA). Add the sample mean of hᵢ/(Nqᵢ) to B. Each sample term is +A or −A and has expectation H. For m independent draws the variance is **(A²−H²)/m**. Uniform draws with the same offset have variance **(mean d²−H²)/m**. Hence proportional sampling saves **Var_pool(|d|)/m** for every possible fixed label vector. If the absolute differences are constant, the designs coincide. If A=0, the difference is zero before any query.

For this estimator the only proposal-dependent part of the one-draw second moment is Σ hᵢ²/(N²qᵢ). Cauchy–Schwarz makes its minimum (Σ |hᵢ|/N)². Hard binary labels imply |hᵢ|=|dᵢ|, so a single label-free proposal attains the bound for every completion. Zero-gap rows contribute exactly zero. A different choice of offset can improve on this particular estimator for some label vectors. For example take f=(1,0), g=(0,1) and y=(0,0). Symmetric centering gives variance 1/m. Moving mean(f²−g²) into the known offset leaves residual −2dy, which is zero on each of these rows. The latter representation remains unbiased for all label vectors.

With fixed labels the number of positive weighted residuals K follows Binomial(m,π), where π is the total proposal mass on positive residuals. Since Δ=B+A(2π−1), transforming a Clopper–Pearson interval gives at least the stated design-based coverage for Δ. A negative upper bound favors f. A positive lower bound favors g. This interval applies to one frozen pair at a fixed number of draws. Repeated indices retain their multiplicity while each label is paid for only once. The expected number of distinct paid labels is Σ[1−(1−qᵢ)ᵐ]. Stopping at a prescribed number of distinct labels creates a different sampling distribution. Repeated checks with optional stopping require a sequentially valid procedure.

## A fixed number of distinct labels

After observing any set S of labels, the **attainable endpoints** are

    lower = B + [Σᵢ∈S hᵢ − Σᵢ∉S |dᵢ|]/N
    upper = B + [Σᵢ∈S hᵢ + Σᵢ∉S |dᵢ|]/N.

Every unopened hard label can attain either sign, so each endpoint is achieved by some completion. With no assumptions about the remaining labels a narrower interval cannot cover all realizations. Its width depends only on the queried indices. Querying the largest remaining |dᵢ| therefore minimizes the width after **every** fixed number of distinct labels. On a particular labeled pool another order can certify the eventual winner sooner. The two sharp endpoints can certify the order as soon as both are strictly positive or both strictly negative. A small true gap can require nearly every informative row.

These conclusions concern the realized finite pool. Predicting future population risk involves variation across inputs and outcomes beyond sampling which labels from this pool to reveal. Full binary Brier summed across both channels doubles the difference and its endpoints, and multiplies its variance by four. Multiclass summed Brier does not generally yield the same two-valued residual.

## Relationship to earlier work

Hansen and Hurwitz (1943) analyzed unequal-probability sampling with replacement from finite populations. Kossen et al. (2021) studied active selection of expensive evaluation labels and corrections for selection bias in model evaluation. The formulas for this paired, symmetrically centered binary Brier contrast and the deterministic completion bounds are derived above. A prior FusionBench experiment estimated target context weights using prediction disagreement and used known labeled source validation subsets to rank candidates. That observer supplies a complementary frozen comparator for a later label-budget study. [Hansen and Hurwitz](https://doi.org/10.1214/aoms/1177731356) · [Kossen et al.](https://proceedings.mlr.press/v139/kossen21a.html) · [Prior FusionBench observer](https://github.com/damiankucharski/FusionBench/blob/1dfb93ee94e837f028ef538795290d57af02fc34/fusionbench/eval/observer.py)

## Independent checks

The auditor accepts a callback for individual labels. It stores the predictions and offsets as rational numbers, uses integer tickets to implement the row probabilities exactly, and computes deterministic completion bounds with rational arithmetic. The binomial interval is evaluated numerically with SciPy. Access tests fail on an unexpected or repeated physical label read, including all-identical forecasts, sampled duplicates, budget exhaustion and early certification. [Access tests](test_design.py)

A second implementation used the three-value rational grid on three rows. Over all 5,832 forecast-pair and label configurations it reproduced the direct empirical risk and the uniform estimator's variance. Its 5,616 positive-mass configurations also reproduced the entire two-draw sign-count distribution. Across all 46,656 combinations with a queried subset, enumeration of every compatible label completion gave both stated endpoints. All 23,328 comparisons of subset widths at each budget agreed with decreasing absolute difference. This is a finite audit of the proof, with zero study labels or model fits. [Independent exact checker](verify.py) · [Recorded counts](verified.json)

```sh
python experiments/paired_label_audit/verify.py
python -m pytest -q experiments/paired_label_audit/test_design.py
```
