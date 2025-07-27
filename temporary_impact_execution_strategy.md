## Part 1: Modeling Temporary Impact

### 1. Data and Sampling

We use full-depth LOB snapshots for three tickers: **CRWV**, **FROG**, and **SOUN**, each spanning **21 trading days**. The raw message-level data is resampled to **1-minute intervals**, yielding **390 snapshots per day**. 

To ensure diversity in **volatility regimes** and **intraday conditions**, we downsample to selected days and key intraday timestamps for model evaluation and visualization.

### 2. Empirical Impact Curves

The empirical impact curves exhibit distinct structural properties: 
- **Flat or concave segments** within individual price level
- **Convex transitions** when crossing to deeper book levels
- **Positive intercepts** due to bid-ask spread

These patterns motivate using models that can capture both **nonlinearity** and **nonzero intercepts**.

 
![Empirical Impact Curve](figures/empirical_impact/empirical_impact_curves.png)

### 3. Model Fitting

We fit the following standard parametric forms to each empirical curve:

- **Linear** (baseline):     $ g(x) = \beta x $
- **Quadratic**:        $ g(x) = \beta_1 x + \beta_2 x^2 $  
- **Power-law**:        $ g(x) = \alpha x^\delta $

![Fitted Impact Curve](figures/fitted_impact/fitted_impact_curves.png)

| Model              | MSE Win Rate | R² Win Rate |
|--------------------|--------------|-------------|
| Power Law          | 83.3%        | 83.3%       |
| Quadratic          | 16.7%        | 16.7%       |
| Linear             | 0.0%         | 0.0%        |

While the standard power law fits well overall, it systematically underestimates impact at small sizes due to its zero-intercept structure. To address this, we introduce the **Shifted Power Law**:

$$
g(x) = \gamma + \alpha x^\delta
$$

where:
- $\gamma:$ intercept term to capture spread-driven slippage
- $\alpha,\delta:$ scale and curvature parameters (as before)

This form maintains the curvature of the power law while allowing for a **non-zero starting point**, better reflecting observed slippage behavior.

We fit the model using non-linear least squares with parameter constraints:
$$
\gamma\geq 0,\quad\alpha\geq 0\quad\delta\geq 0
$$

Initial values:
- $\gamma=\min g(x)$
- $\alpha = 1.0,\quad\delta=1.0$

![Shifted Impact Curve](figures/shifted_impact/shifted_impact_curves.png)

| Model              | MSE Win Rate | R² Win Rate |
|--------------------|--------------|-------------|
| Shifted Power Law  | 88.6%        | 87.6%       |
| Quadratic          | 8.1%         | 8.1%        |
| Power Law          | 3.3%         | 4.3%        |
| Linear             | 0.0%         | 0.0%        |

The Shifted Power Law significantly outperform Standard Power Law. It corrects underfitting at small sizes while preserving the realistic curvature observed in empirical data.

This model will serve as the functional form for optimization in Part 2.

---

## Part 2: Mathematic Framework & Execution Algorithm

### 1. Problem Formulation
The objective is to minimize cumulative **temporary impact**, while ensuring complete execution:

$$
\min_{x_1, \dots, x_N} \sum_{t=1}^{N} g_t(x_t),\quad \text{s.t.} \sum_{t=1}^{N} x_t = S,\quad x_t \geq 0
$$

The impact function $g_t(x_t)$ represents the **execution cost** of trading $ x_t $ shares at time $ t $. We model this using a **shifted power-law** function consistent with observed limit order book behavior:

$$
g_t(x_t) = \gamma_t + \alpha_t x_t^{\delta_t}
$$

where:
- $ \gamma_t \geq 0: $ fixed cost paremeter (ideally half the bid-ask spread)
- $ \alpha_t > 0: $ scale parameter (sensitivity to order size)
- $ \delta_t > 0: $ curvature parameter (degree of nonlinearity in impact) 

Since the future impact functions $ \{g_t\}_{t=1}^N $ is unknown in advance, we adopt an online execution framework where each allocation $x_t$, uses only past and current data. Slippage parameters are estimated in real time, allowing the strategy to adapt dynamically to evolving market conditions.

### 2. Parameter Estimation

At each time $ t $, the model parameters $ \theta_t = (\gamma_t, \alpha_t, \delta_t) $ must be estimated in real time. We construct these estimates by combining two sources of information:

1. **Recent market conditions**, using the most recent $ W $ minutes within the current day  
2. **Historical time-of-day behavior**, using the same minute $ t $ from the past $ D $ trading days

Both components are computed using **exponentially decaying weights** to emphasize recency within their respective windows.

Let:
- $ \theta_{t-m}^{(m)} $: parameter estimate from minute $ t - m $ within the current day, where $ m = 0, \dots, W - 1 $
- $ \theta_t^{(d)} $: parameter estimate at time $ t $ from trading day $ d $, where $ d = 1 $ is the most recent day and $ d = D $ is the furthest

**The recent estimate** is defined as:

$$
\theta_t^{\text{recent}} = \sum_{m=0}^{W-1} w_m^{\text{min}} \cdot \theta_{t - m}^{(m)}, \quad
w_m^{\text{min}} = \frac{\rho_{\text{min}}^m}{\sum_{j=0}^{W-1} \rho_{\text{min}}^j}
$$

**The historical estimate** is defined as:

$$
\theta_t^{\text{historical}} = \sum_{d=1}^{D} w_d^{\text{day}} \cdot \theta_t^{(d)}, \quad
w_d^{\text{day}} = \frac{\rho_{\text{day}}^{d-1}}{\sum_{j=1}^{D} \rho_{\text{day}}^{j-1}}
$$

where $ \rho_{\text{min}}, \rho_{\text{day}} \in (0, 1) $ are **exponential decay rates** for recent minutes and past days, respectively.

**The final parameter estimate** for each future period $ i \geq t $ is a combination of two components:

$$
\hat{\theta}_i = (1 - \lambda_i) \cdot \theta_i^{\text{historical}} + \lambda_i \cdot \theta_t^{\text{recent}}, \quad \lambda_i = \lambda \cdot \left( \frac{N - i + 1}{N - t + 1} \right)
$$

where $ \lambda \in [0, 1] $ controls the overall reliance on recent versus historical information, while $ \lambda_i $ applies a **time-decaying weight** to reduce the influence of recent microstructure estimates as the forecast horizon extends.

This framework enables short-term responsiveness while preserving long-term stability.

### 3. Dynamic Allocation Strategy

At each time step \( t \), the goal is to allocate a portion \( x_t \) of the remaining order volume \( R_t = S - \sum_{i=1}^{t-1} x_i \) in order to minimize total slippage over the rest of the trading day.

Using the parameter estimation procedure described in Section 2, we compute forecasts \( \hat{\theta}_i = (\hat{\gamma}_i, \hat{\alpha}_i, \hat{\delta}_i) \) for each future period \( i \in \{t, \dots, N\} \).

#### Cost-Aware Weighting

To allocate volume efficiently, we define a cost-aware weight for each future period \( i \geq t \):

\[
w_i = \left( \hat{\alpha}_i^{1/\hat{\delta}_i} + \lambda_\gamma \cdot \hat{\gamma}_i \right)^{-1}
\]

where:
- \( \hat{\alpha}_i, \hat{\delta}_i, \hat{\gamma}_i \) are parameters of the shifted power-law impact model at time \( i \),
- \( \lambda_\gamma \geq 0 \) is a hyperparameter that controls the penalty on fixed costs (e.g., spread).

This weighting balances two components:
- \( \hat{\alpha}_i^{1/\hat{\delta}_i} \): a proxy for expected marginal cost,
- \( \hat{\gamma}_i \): a proxy for fixed cost (e.g., initial slippage for small orders).

As \( \lambda_\gamma \) increases, the strategy defers volume away from periods with high fixed cost.

#### Volume Allocation Rule

The allocation for the current time step \( x_t \) is then computed as:

\[
x_t = \frac{w_t}{\sum_{i = t}^{N} w_i} \cdot R_t
\]

This ensures that:
- Remaining volume \( R_t \) is proportionally distributed based on forecasted execution costs,
- The total constraint \( \sum_{t=1}^N x_t = S \) is satisfied by construction,
- The strategy adapts dynamically to evolving market conditions and real-time signal updates.

This rule is applied recursively at each time step, forming a fully online and adaptive execution strategy.

---

## Appendix
### A. Empirical Impact Curves
![Empirical Impact Curve](figures/empirical_impact/crwv_empirical_impact.png)
![Empirical Impact Curve](figures/empirical_impact/frog_empirical_impact.png)
![Empirical Impact Curve](figures/empirical_impact/soun_empirical_impact.png)

### B. Fitted Model Comparison
![Fitted Impact Curve](figures/fitted_impact/crwv_fitted_impact.png)
![Fitted Impact Curve](figures/fitted_impact/frog_fitted_impact.png)
![Fitted Impact Curve](figures/fitted_impact/soun_fitted_impact.png)

### C. Shifted vs. Standard Power-Law Comparison
![Shifted Impact Curve](figures/shifted_impact/crwv_shifted_impact.png)
![Shifted Impact Curve](figures/shifted_impact/frog_shifted_impact.png)
![Shifted Impact Curve](figures/shifted_impact/soun_shifted_impact.png)