# Companion: Numerical $P(m)$ and $P(w)$ under logit-normal conditional dispersion

> **Self-contained brief for a fresh session.** This document plans how to keep numerical predictions for the balance distribution $P(m)$ and the transaction-size distribution $P(w)$ when the conditional dispersion of $q = w/m$ is logit-normal rather than Beta-Binomial. The closed-form NB chain in the existing SM breaks; the variance decomposition does not. We replace the broken parts with 1D numerical integrals and validate against agent-based simulation.

## 0. Pointers (read these first)

- **Main paper**: `paper/Transactions/SUBMISSION_PRL.tex`
- **SM** (the math we're modifying): `paper/Transactions/SM.tex` — the relevant subsections are §B.1 (Binomial $p(m\mid s)$), §B.2 (Beta-Binomial $p(m\mid s)$ and $r_\mathrm{eff}$), §C (the $P(w)$ derivations)
- **Empirical motivation**: F2c–F2g live in `berka/dispersion_alternatives.ipynb` (companion notebook; not `empirical_analysis.ipynb`). F2c (BetaBin $\hat\xi$ moment-match), F2f (per-distribution fits incl. $\hat\tau_\mathrm{logit{-}N}$), F2g (per-account variance scaling — empirical free-slope $\alpha \approx 1.87$ for all debits, 1.64 for voluntary; Beta predicts slope 1, logit-normal slope 2). Aggregate Berka calibration numbers (mean balance, mean tx, $h(s)$ bounds) are in §5 of `berka/empirical_analysis.ipynb`.
- **Generative model**: `src/model.py`, `src/dists.py`, `src/simulations.py` — the existing simulator (BetaBin-based). The `logitnormal-numerics` branch added a parallel `dispersion='logitnormal'` option in `pay_random_share_logitn` + dispatch in `transact` + plumbing in `src/execution.py`.
- **Numerical pipeline (post-reorg layout)**: distribution math lives under `numerical/` (not `src/`). The BetaBin/NB sibling is `numerical/distributions.py`; the logit-normal modules added on this branch are `numerical/logitnormal.py` (Poisson-LogitNormal conditional pmf) and `numerical/numerical_pm_pw.py` (outer integrals). The Berka-vs-numerical overlay notebook is `berka/numerical_validation.ipynb`.

## 1. The substitution

**Old (Beta-Binomial):** $w \mid M, s \sim \mathrm{BetaBin}(M, \xi s, \xi(1-s))$, giving
$$\mathrm{Var}(w\mid M, s) = M s(1-s)\Big[1 + \tfrac{M-1}{\xi+1}\Big] \;\xrightarrow{M\to\infty}\; M s(1-s) + \tfrac{(Ms)^2(1-s)}{s(\xi+1)}.$$

**New (Logit-normal):** $\mathrm{logit}(q) \sim \mathcal{N}(\mathrm{logit}\,s,\, \tau^2)$, with $w\mid M, q \sim \mathrm{Binomial}(M, q)$. By delta method,
$$\mathrm{Var}(q\mid s, \tau) \approx \tau^2 [s(1-s)]^2, \qquad \mathrm{Var}(w\mid M, s, \tau) \approx M s(1-s) + M^2 \tau^2 [s(1-s)]^2.$$

The crucial observation: in the small-$s$ large-$M$ limit, the overdispersion contribution is $\sigma^2_\tau \approx \lambda_s^2\,\tau^2$ — **independent of $s$**. This is what makes the variance-decomposition algebra cleaner than BetaBin's, where $\sigma^2_\xi \propto \lambda_s^2/[s(\xi+1)]$ silently rolled an $s$-dependence into ξ.

## 2. What survives, what breaks

| SM section | Equation(s) | Status under logit-normal | Replacement |
|---|---|---|---|
| §B.1 Binomial $p(m\mid s)$ | Eq. (18) Poisson | unchanged | — |
| §B.2 BetaBin $p(m\mid s)$ | Eq. (29) NB form | **broken** — no NB limit | numerical Poisson-logit-normal compound (§3) |
| §B.2 Variance decomp | Eq. for $\sigma^2_\mathrm{eff}$ | survives, *cleaner* | $1/r_\mathrm{eff} = 1/r + \tau^2$ exact in small-$s$ limit |
| §B.2 Heavy tail of $P(m)$ | Eq. for $P(m)\sim m^{-2}$ | likely survives | re-derive from Poisson-LogN mixture; tail driven by $h(s)$ heterogeneity, not the conditional shape |
| §C $P(w)$ via NB compounding | Eq. (43) and surrounding | **broken** — no NB | Poisson-LogN compound (§4) |
| §C asymptotic Poisson tail | Eq. for $P(w)\sim w^{-?}$ | needs re-derivation | saddlepoint or direct numerics on PLN |
| Master eq. & stationarity | §A | unchanged (model-agnostic) | — |
| Stat. probability $\pi_{a,b,s}$ | Eq. (12) | unchanged | — |

Net: the architecture (master equation → stationary $\pi_{a,b,s}$ → conditional $p(m\mid s)$ → integrate over $h(s)$) is preserved. The conditional shape changes, and we lose the NB algebra in exchange for 1-D numerical quadrature.

## 3. Numerical $P(m)$

### 3.1 Conditional $p(m \mid s, \tau)$

Under logit-normal jumps, the equilibrium $p(m\mid s)$ is the marginal of a Poisson-rate compound: at large $M$, an activation event sends $\sim \mathrm{Poisson}(M q)$ of the $m$ resident RWs, with $q\sim \mathrm{LogitN}(\mathrm{logit}\,s,\, \tau^2)$. The stationary balance distribution given $s$ is the **Poisson-LogitNormal compound**:

$$p(m\mid s, \tau) \;=\; \int_0^1 \mathrm{Poisson}(m;\, \lambda(q,s)) \cdot \mathrm{LogitN}(q;\, s, \tau^2)\, dq.$$

Here $\lambda(q,s) = \lambda_s \cdot q/s$ in the small-$s$ approximation (rate scales linearly with the realized jump probability over its mean). Equivalently, change variables to $z = \mathrm{logit}(q)$:

$$p(m\mid s, \tau) \;=\; \int_{-\infty}^\infty \mathrm{Poisson}\!\Big(m;\, \lambda_s\,\tfrac{\sigma(z)}{s}\Big) \cdot \frac{1}{\tau\sqrt{2\pi}}\, e^{-\frac{(z - \mathrm{logit}\,s)^2}{2\tau^2}}\, dz$$

where $\sigma(z) = (1+e^{-z})^{-1}$.

**Numerical recipe**: Gauss-Hermite quadrature in $z$ (15-30 nodes is plenty), evaluating Poisson pmf at each node. Cost: $O(|m\text{-grid}| \cdot |z\text{-grid}|)$, i.e. cheap.

### 3.2 Marginal $P(m)$

$$P(m) \;=\; \int_{s_\min}^{s_\max} p(m\mid s, \tau)\, h(s)\, ds$$

with $h(s) \sim s^\sigma$ as in the existing SM (§B.2). Use Gauss-Legendre on $\log s$ (50-100 nodes), since $s$ spans 2-3 orders of magnitude in the calibrated case.

### 3.3 Heavy-tail derivation (replacement for §B.2)

The paper's $P(m)\sim m^{-2}$ result follows from $\int s^\sigma \, m^{-r/s}\, ds$-type asymptotics where the NB conditional gave the $m^{-r/s}$ factor. Under PLN, the conditional has a different (heavier, log-normal-tinted) shape but the tail is still controlled by integration over $h(s)$ for small $s$. Sketch:

- For small $s$ (and given $\tau$), the PLN conditional is approximately $\mathrm{LogN}(\mathrm{logit\text{-}}\mu, \tau^2)$-rate Poisson, whose tail at large $m$ is $\sim e^{-(\log m - \log \lambda_s)^2/(2\tau^2)}$ (log-normal tail behavior, sub-exponential).
- Integrating against $h(s)\sim s^\sigma$, the exponential of the log-normal still allows a power-law to emerge from the mixture — but the exponent is *not* generally $-2$; it depends on $\tau$ and $\sigma$.
- **Action item**: derive the analog of Eq. (29) explicitly. Saddle-point on the integral over $s$ should give the new exponent in closed form.

## 4. Numerical $P(w)$

### 4.1 Conditional $p(w \mid m, s, \tau)$

A node with $m$ resident RWs and spending propensity $s$ activates and a fresh
realized jump fraction $q' \sim \mathrm{LogitN}(\mathrm{logit}\,s, \tau^2)$ is
drawn for that activation event. The number of RWs sent is $\mathrm{Bin}(m, q')$,
which for large $m$ is approximately $\mathrm{Poisson}(m q')$. Compounded over $q'$:

$$p(w \mid m, s, \tau) \;=\; \int_0^1 \mathrm{Poisson}(w;\, m q') \cdot \mathrm{LogitN}(q';\, s, \tau^2)\, dq'.$$

This is a Poisson-LogitNormal pmf at fixed scale $m$ — implemented in
[`log_pw_cond_pln`](../../numerical/logitnormal.py).

### 4.2 Marginal $P(w)$

Following SM eq. 624, sum over $m$ and integrate over $h(s)$:

$$P(w) \;=\; \sum_m \int_{s_\min}^{s_\max} p(w \mid m, s, \tau)\, p(m \mid s, \tau)\, h(s)\, ds$$

with $p(m \mid s, \tau)$ the equilibrium occupancy of §3.1. Both inner factors
are Poisson-LogitNormal compounds at different scales ($\lambda_s q/s$ for $m$,
and $m q'$ for $w \mid m$, with $q$ and $q'$ independent draws from the same
LogitN). The chain doesn't admit a clean closed form like the BetaBin
NB$\to$NB thinning identity, so we evaluate by Monte Carlo over
$(s, q, m, q', w)$ — see [`Pw_logitn_mc`](../../numerical/numerical_pm_pw.py).
With $10^6$ samples the noise on log-binned $P(w)$ is sub-1% across $\sim 8$
decades.

### 4.3 Cross-check: PLN as the "Berka-calibrated" alternative to NB

For Sarafu the paper found $\xi=1$. For Berka the empirical-driven τ is:

| subset | $\hat\tau$ (F2f MLE) | $\hat\xi$ (F2c moment) | empirical free-slope α |
|---|---:|---:|---:|
| all debits | 1.166 | 6.20 | 1.87 |
| voluntary  | 0.875 | 9.26 | 1.64 |

The free-slope $\alpha$ from F2g sits between Beta's predicted slope $1$ and
LogitN's slope $2$, suggesting that neither shape is exactly right; an
intermediate-tail family (e.g. truncated normal or generalized normal on the
logit scale) might fit the per-account variance scaling more cleanly. See §7
open question.

Use the voluntary calibration as the working operating point for $P(m)$ and
$P(w)$ comparisons (cleanest behavioural subset; "all debits" mixes in
mortgage/loan-payment patterns that don't fit a proportional-spend model).

## 5. Implementation map

The logit-normal pipeline lives alongside the BetaBin pipeline in `numerical/`,
with a parallel `dispersion='logitnormal'` option in the agent simulator under
`src/`. All implemented on the `logitnormal-numerics` branch.

| Concept | BetaBin/NB (existing) | LogitN (this branch) |
|---|---|---|
| Conditional $p(q\mid s)$ | $\mathrm{Beta}(\xi s, \xi(1-s))$ | $\mathrm{LogitN}(\mathrm{logit}\,s, \tau^2)$ |
| Conditional $p(m\mid s)$ | `Pm_single` (NB) | [`Pm_single_logitn`](../../numerical/numerical_pm_pw.py) (PLN compound) |
| Marginal $P(m)$ | `compute_Pm` | [`compute_Pm_logitn`](../../numerical/numerical_pm_pw.py) |
| Conditional $p(w\mid m, s)$ | `betabinom.pmf(m, ξs, ξ(1-s))` | [`log_pw_cond_pln`](../../numerical/logitnormal.py) (PLN at scale $m$) |
| Marginal $P(w)$ (SM eq. 624) | `Pw_single_local` (Poisson-Gamma-Beta) | [`Pw_logitn_mc`](../../numerical/numerical_pm_pw.py) (Monte Carlo) |
| Simulator option | default | `dispersion='logitnormal', tau=...` in [`src/model.py:transact`](../../src/model.py) |

Quadrature note: the inner LogitN integral (in `log_pw_cond_pln` /
`log_pm_cond_pln`) uses a fine trapezoidal rule in the standard-normal residual
$z \in [\pm 8]$ at $n_z = 400$. Gauss-Hermite at modest $n$ aliases for $\tau \sim 1$.

## 6. Validation

Worked through in [`berka/numerical_validation.ipynb`](../../berka/numerical_validation.ipynb):

1. **$\tau \to 0$ collapse**: PLN conditional pmf reduces to $\mathrm{Poisson}(\lambda)$ at machine precision (max abs diff $\sim 10^{-14}$).
2. **Conditional density** $p(q\mid s)$: BetaBin and LogitN side-by-side at three $s$ values, plus simulator-emitted $q = w/\mathrm{src\_bal}$ overlay. Both options' simulator output sits on the analytic density across log-binned tails.
3. **Conditional occupancy** $p(m\mid s)$: numerical curves vs simulator-binned histograms. Both options consistent with simulator at fixed $s$.
4. **Marginal** $P(m)$, $P(w)$: numerical compounding (NB closed form for BetaBin, MC for LogitN) tracks simulator histograms within MC noise across 8+ decades.
5. **Berka empirical** $P(m)$, $P(w)$ at the calibrated $(\hat\xi, \hat\tau)$: model captures the bulk; small-$w$ tail (Berka has $w < 10$ Kč mass that neither model produces — bank fees / fixed-charge transfers) is real model-vs-data residual.
6. **Data-driven** $P(w)$ (notebook §B.3): bypasses $h(s) \propto s^\sigma$ by drawing $q$ per-tx from each option using the *empirical* per-account $(s_i, m_i)$. Isolates the dispersion choice as the only model knob. At voluntary calibration: BetaBin median tracks empirical to within ~6%; LogitN overshoots by ~30% (its heavier large-$q$ tail at calibrated $\hat\tau$ pushes per-tx amounts higher than observed).

## 7. Open questions / risks

- **Tail exponent**: do we recover $P(m)\sim m^{-2}$ under logit-normal? Probably no — needs the saddle-point derivation in §3.3. If the new exponent depends on $\tau$, that's a feature (lets $\tau$ explain dataset-to-dataset tail variation) but it complicates the universality story.
- **Co-jumping correlation $r$**: the paper introduced $r$ for co-location of RWs. Under logit-normal, the corresponding $r_\mathrm{eff} = 1/(1/r + \tau^2)$. The interpretation of $r$ itself doesn't change. But: do we need to revisit the derivation of $r$ from the SM in §B.2 footnote? It used BetaBin's intra-class-correlation; logit-normal's correlation structure is different (Gaussian on logit scale). Worth a careful read.
- **Boundary effects**: logit-normal has well-behaved tails on $[0,1]$ but $\mathrm{logit}\,s_\min$ blows up if $s_\min \to 0$. The paper sets $s_\min = s_\max/N$; for very large $N$ this could create numerical issues. Use float64 throughout and clip $|z| < 30$ in the Hermite quadrature.
- **Empirical α not exactly 2**: the free-slope fit gave 1.64 / 1.87, between Beta and logit-normal. Pure logit-normal will overshoot the true scaling somewhat. If this matters for the paper's quantitative match, the alternative is a *generalized normal* (Subbotin) on the logit scale with shape parameter $p$ fit empirically. That introduces one more parameter; defer unless needed.

## 8. Deliverables — status

**Done on the `logitnormal-numerics` branch:**

1. **`numerical/logitnormal.py`** — Poisson-LogitNormal conditional pmf via fine trapezoidal quadrature on the logit residual. Helpers `logitn_q_quadrature`, `log_pln_compound`, `log_pm_cond_pln`, `log_pw_cond_pln`. $\tau \to 0$ collapses to Poisson at machine precision.
2. **`numerical/numerical_pm_pw.py`** — `Pm_single_logitn` and `compute_Pm_logitn` for $P(m)$ (outer log-spaced trapezoid in $s$, reusing `log_h` / `_integrate_logspace` / `compute_params` from `numerical/distributions.py`); `Pw_logitn_mc` for $P(w)$ via the SM-style compounding chain.
3. **Simulator option** `dispersion='logitnormal'` — `src/model.py:pay_random_share_logitn`, dispatched in `transact` via the new `dispersion` and `tau` kwargs; plumbed through `src/execution.py:run_simulation` and into the metadata dict.
4. **Validation notebook** — `berka/numerical_validation.ipynb` works through §A (numerical vs simulator agreement, both options) and §B (Berka empirical overlay + dispersion-only data-driven comparison).

**Deferred to follow-up:**

- **Tail asymptotic for $P(m)$ under PLN** (§3.3): saddle-point on the integral over $h(s)$ to recover the analog of $P(m)\sim m^{-2}$. Tail exponent likely depends on $\tau$ and $\sigma$.
- **SM §B.2 / §C replacement section** describing the logit-normal route. Keep the BetaBin section for comparison; the contribution becomes "model-agnostic up to choice of conditional dispersion; here we present results for both."
- **Co-jumping correlation $r$** under logit-normal: the identity $1/r_\mathrm{eff} = 1/r + \tau^2$ is used in code but not re-derived from first principles. The paper's $r$ derivation in SM §B.2 footnote uses BetaBin's intra-class correlation; the logit-normal counterpart is Gaussian on the logit scale.
- **Empirical $\alpha \approx 1.64$** (voluntary, F2g): between Beta's slope $1$ and LogitN's slope $2$. A truncated normal or generalized normal (Subbotin) on the logit scale with shape parameter fit empirically might fit the per-account variance scaling more cleanly. Defer until needed for paper match.
- **Nomenclature tidy** — the simulator's `pay_random_share` arg named `p` is the SM's $s$, and the kwarg named `s` is the SM's $\xi$. Cross-cutting rename; pre-existing in `main`.

## 9. Things to NOT do in the next session

- Don't re-fit $\hat\tau$ from the residual histograms — the F2f MLE values in §4.3 are stable.
- Don't re-do the empirical analysis in `berka/empirical_analysis.ipynb` or the candidate-shape comparison in `berka/dispersion_alternatives.ipynb` — both stable.
- Don't extend `src/model.py` to *replace* the BetaBin simulator. Logit-normal is added as a parallel option (`dispersion='logitnormal'`) alongside it; paper likely presents both.
