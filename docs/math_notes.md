# Mathematical Notes

Concise derivations behind the modeling choices. Notation: $y_i \in \{0,1\}$
(churn), $\mathbf{x}_i$ features, $p_i = P(y_i=1 \mid \mathbf{x}_i)$.

## 1. Logistic loss (binary cross-entropy)

Model $p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$, $\sigma(z) = 1/(1+e^{-z})$.
The negative log-likelihood over $n$ i.i.d. observations is

$$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^n \big[ y_i \log p_i + (1-y_i)\log(1-p_i) \big].$$

It is convex in $\mathbf{w}$ (the Hessian $\mathbf{X}^\top \mathbf{D} \mathbf{X}$
with $D_{ii} = p_i(1-p_i) \succeq 0$), so gradient methods converge to the global
minimum. `class_weight="balanced"` multiplies each term by
$n/(2\,n_{\text{class}})$, i.e. it re-weights the empirical risk so the minority
class contributes half the total loss.

## 2. Gradient boosting (additive Newton steps)

Boosting builds $F_M(\mathbf{x}) = \sum_{m=1}^M \gamma_m h_m(\mathbf{x})$
greedily. At round $m$, with current predictions $F_{m-1}$, XGBoost minimizes the
second-order Taylor expansion of the loss:

$$\sum_i \big[ g_i h(\mathbf{x}_i) + \tfrac{1}{2} h_i h(\mathbf{x}_i)^2 \big]
+ \Omega(h), \qquad
g_i = \frac{\partial \ell}{\partial F_{m-1}}, \;
h_i = \frac{\partial^2 \ell}{\partial F_{m-1}^2}.$$

For logistic loss, $g_i = p_i - y_i$ (residual) and $h_i = p_i(1-p_i)$.
The optimal leaf weight is $w_j^* = -G_j / (H_j + \lambda)$ with
$G_j = \sum_{i \in j} g_i$, $H_j = \sum_{i \in j} h_i$ — a Newton step per leaf,
regularized by $\lambda$. `scale_pos_weight` multiplies $g_i, h_i$ of positive
samples, shifting every split's gain toward separating churners.

## 3. Calibration

A model is calibrated if $P(y=1 \mid \hat p = q) \approx q$ for all $q$.
Isotonic regression finds the non-decreasing function $m$ minimizing
$\sum_i (m(\hat p_i) - y_i)^2$ (PAVA algorithm) — non-parametric, so it corrects
arbitrary miscalibration shapes. Calibration quality is measured by the
**Brier score** $\frac{1}{n}\sum_i (\hat p_i - y_i)^2$, a *proper scoring rule*:
it is minimized in expectation only by reporting true beliefs, so improving it
cannot be gamed.

## 4. Expected-value thresholding

Let offer cost $c$, save probability $s$, retained-customer value $V_i$.
Contacting customer $i$ has expected value

$$\mathbb{E}[\text{value}_i] = p_i \cdot s \cdot V_i - c,$$

positive iff $p_i > c/(s V_i)$. With per-customer $V_i = 6 \times \text{MonthlyCharges}_i$,
the break-even probability differs per customer; we therefore sweep a *global*
threshold on validation data and pick the maximizer of total expected value.
This replaces the arbitrary 0.5 cutoff with a decision grounded in the cost of
being wrong — and it is only valid because the probabilities are calibrated (§3).

## References

- Friedman, J. (2001). Greedy function approximation: a gradient boosting machine.
- Chen & Guestrin (2016). XGBoost: A scalable tree boosting system.
- Niculescu-Mizil & Caruana (2005). Predicting good probabilities with supervised learning.
- Zadrozny & Elkan (2002). Transforming classifier scores into accurate multiclass probability estimates.
