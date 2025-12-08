# grok-scientific.md: Scientific Document for Distributional Supervised Learning on NASDAQ Futures

## 1. Executive Summary & Scientific Rationale

This document articulates the scientific ideation for a supervised learning model predicting NASDAQ 100 (NQ) futures returns over horizons of 15m, 30m, 60m, 2h, and 4h. Prioritizing accuracy and precision, we conceptualize a distributional forecasting approach using a hybrid Transformer architecture. Grounded in the research guide's synthesis of financial data mechanics, variable embeddings, gated normalization, and distributional outputs.

The core hypothesis is that NQ futures exhibit non-stationary, multivariate dependencies with heavy-tailed noise. Distributional outputs capture uncertainty better than point predictions. By embedding variables rather than timesteps and applying gated normalization, the model disentangles temporal and cross-variable dynamics, enhancing signal extraction in noisy markets.

This ideation integrates RL-TVDT (variable and temporal attention), MIGT (gated instance normalization), and broader multivariate time series literature from the research guide.

## 2. Ideation of Market Behavior

NQ futures aggregate expectations of technology equities, influenced by macroeconomic news, liquidity flows, and microstructure. Key behaviors:

- **Non-Stationarity (H1):** Statistical properties shift due to regimes (e.g., low-vol 2010s vs. high-vol post-2020). Models must learn shape-invariant patterns via per-instance normalization.
- **Multivariate Dependencies (H2):** Features like price, volume, RSI interact dynamically—e.g., volume amplifies momentum but signals reversals in illiquidity. Variable-centric attention reveals these.
- **Heavy-Tailed Returns (H3):** Leptokurtic distributions from tail events. Distributional modeling captures asymmetry and uncertainty.
- **Cyclical Patterns (H4):** Intraday (open/close volatility), weekly (Monday effects), seasonal (quarterly flows) drive predictability. Composite embeddings learn these.
- **Noise Dominance (H5):** 5-min bars include bid-ask noise; gating suppresses it.
- **Heterogeneous Time Scales (H6):** Microstructure (short bars) vs. intraday/daily trends require factorized attention.
- **Regime-Dependent Predictability (H7):** Volatility regimes modulate uncertainty; wide quantile spreads signal "stay out."

Hypothesis: Variable embeddings + two-stage attention model causal mechanisms (e.g., volume leading price) better than timestep embeddings.

## 3. Mathematical Foundations

Let input be multivariate time series \( X_t \in \mathbb{R}^{T \times V} \), where \( T \in [273, 276] \) (lookback bars), \( V \) (variables: OHLCV + derived).

For horizon \( h \in \{15, 30, 60, 120, 240\} \) minutes, predict conditional distribution \( \hat{F}_{t,h}(r | X_t) \) of log-return \( r_{t,h} = \ln(P_{t+h} / P_t) \).

### 3.1. Input Representation & Handling Irregularities
Pad to \( T_{max} = 288 \) with zeros; use mask \( M \in \{0,1\}^{T_{max}} \) (1 for valid). Attention: \( \text{softmax}(QK^T / \sqrt{D} + (1-M) \cdot (-\infty)) V \).

Ratio back-adjustment preserves log-returns: \( P_{adj,t} = P_{raw,t} \times \prod_{rolls} R_i \), where \( R_i = P_{new,i} / P_{old,i} \).

Embed each timestep: Project \( X_t \) to \( E \in \mathbb{R}^{T \times V \times D} \) via linear layer per variable, preserving time dimension for temporal attention.

Positional encodings (added to \( E \) before temporal attention):
- Time of day: \( PE_{tod}(t) = [\sin(2\pi t / 288), \cos(2\pi t / 288)] \), applied per timestep.
- Day of week: Learnable \( E_{dow} \in \mathbb{R}^D \), broadcast across time.
- Day of month/year: Time2Vec \( T2V(\tau)[0] = \omega_0 \tau + \phi_0 \), \( T2V(\tau)[i] = \sin(\omega_i \tau + \phi_i) \) for \( i \geq 1 \), with learned \( \omega, \phi \); applied per timestep to capture long-range cycles.

Horizon embedding: For multi-horizon output, embed horizon index as learnable vector \( E_h \in \mathbb{R}^D \), concatenated or added to decoder inputs for shared representations across horizons.

### 3.2. Derived Features
Features capture momentum, volatility, liquidity, trend (computed causally to avoid lookahead bias—e.g., rolling stats use only past data):

- **Volatility:** Garman-Klass \( \sigma^2 = 0.5 (\ln H - \ln L)^2 - (2\ln 2 - 1)(\ln C - \ln O)^2 \); rolling realized \( RV_n = \sqrt{\sum_{i=t-n}^{t-1} r_i^2} \) for \( n \in \{12, 36, 72\} \) (note: Parkinson omitted due to high correlation >0.95 with Garman-Klass to avoid multicollinearity redundancy).
- **Liquidity:** Amihud \( ILLIQ_t = |r_t| / (P_t V_t) \).
- **Momentum:** RSI \( = 100 - 100 / (1 + RS) \), \( RS = \bar{U}/\bar{D} \); MACD = EMA_{12} - EMA_{26}; ROC_n = (C_t - C_{t-n}) / C_{t-n}.
- **Trend:** EMA slope \( \Delta EMA_n = EMA_n(t) - EMA_n(t-1) \) for \( n \in \{9, 21, 50\} \); deviation \( (C_t - EMA_n)/EMA_n \).
- **Range:** ATR \( TR_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|) \), rolling average.

Log-volume and standardized returns handle scale.

## 4. Algorithms & Architecture Concept

### 4.1. Hybrid MIGT-TVDT Concept
Invert Transformer: Process per-variable series with temporal attention first, then aggregate and apply variable attention, with gating for stability.

- **Variable Embedding:** For each variable \( v \), embed the full time series: \( E_v \in \mathbb{R}^{T \times D} \) via linear projection + positional encodings (preserving time dimension).
- **Two-Stage Attention (TSA):**
  1. Temporal: Per variable \( v \), self-attention over time dimension: \( Z_v = \text{Attention}(E_v W_Q^{temp}, E_v W_K^{temp}, E_v W_V^{temp}) \in \mathbb{R}^{T \times D} \); aggregate \( \bar{Z_v} \in \mathbb{R}^D \) via attention-weighted pooling (learnable query attends to all \( Z_v \) timesteps bidirectionally, as the lookback is fully historical and causality is not strictly required for aggregation—preserves relative sequence importance without obscuring order).
  2. Variable: Stack \( \bar{Z} \in \mathbb{R}^{V \times D} \); apply attention across variables: \( H = \text{Attention}(\bar{Z} W_Q^{var}, \bar{Z} W_K^{var}, \bar{Z} W_V^{var}) \); learns inter-variable correlations (e.g., RSI-Volume).
- **Gated Instance Normalization:** Normalize per window: \( \hat{x} = (x - \mu) / \sigma \); apply post-attention. Lite Gate Unit (LGU): \( G = \sigma(W \hat{x} + b) \), then residual output \( x_{out} = x + G \odot \text{Attention}(\hat{x}) \). This matches MIGT's post-attention gating to filter noisy updates and adapt to non-stationarity.
- **Reversible Instance Norm (RevIN):** Apply to raw input \( X_t \) before embedding (normalize to zero mean/unit variance, store stats); reverse after final decoder to restore original scale.

Stack layers; shared encoder with horizon-specific decoder heads (e.g., MLPs per \( h \), conditioned on \( E_h \)).

### 4.2. Distributional Approaches
Output distribution per horizon.

- **Primary: Quantile Regression.** Predict \( \hat{q}_\tau \) for \( \tau \in \{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95\} \). Loss: Pinball \( L_\tau(y, \hat{q}_\tau) = \tau (y - \hat{q}_\tau)_+ + (1-\tau) (\hat{q}_\tau - y)_+ \). Prevent crossing: \( \hat{q}_\tau = \hat{q}_{min} + \sum \text{softplus}(\delta_i) \).
- **Alternative: Categorical.** Bin returns (e.g., 51 bins); predict probabilities via softmax.

Hypothesis: Quantiles yield calibrated precision (interval width as uncertainty), superior in tails.

## 5. Hypotheses & Testable Predictions

- **H8:** TSA + variable embeddings improve IC over standard Transformers by modeling feature interactions.
- **H9:** Gated normalization enhances cross-regime generalization, reducing variance in volatile periods.
- **H10:** Quantiles achieve lower CRPS than MSE, with coverage ≈ nominal (e.g., 80% for [0.1, 0.9]).
- **H11:** Composite embeddings capture cycles, boosting performance on Fridays vs. Mondays.

Test via ablations (on validation set):

| Ablation | Modification | Expected Outcome |
|----------|--------------|------------------|
| No Variable Embedding | Standard timestep tokens | Lower IC, entangled correlations |
| No TSA | Single-stage attention | Higher compute, missed intra-variable dynamics |
| No Instance Norm | LayerNorm only | Poor regime adaptation (e.g., 2024+ data) |
| No LGU | Remove gating | Noisier predictions, wider intervals |
| Mean Pool vs. Weighted | Use mean in aggregation | Obscured sequence importance, lower tail calibration |
| Point Prediction | MSE loss | No uncertainty, similar median but worse CRPS |

Success: Gross Sharpe >1.0 (pre-costs) in simulated strategy (long/short on median >θ, sized inversely to interval width); target >1.5 ambitious without costs.

## 6. Training Strategy

- **Data Splits:** Train: 2010-2021 (diverse regimes); Val: 2022-2023 (bear market); Test: 2024-Dec 2025 (bull market). Strict time-based to prevent lookahead.
- **Optimizer:** AdamW, LR 1e-4 base; cosine schedule with warmup (1000 steps) to 1e-6 min.
- **Hyperparameters:** \( D \in [128, 512] \), heads 4-8 (scaled to \( V \approx 20-30 \)); batch 64-256 (A100 fits ~10M params comfortably, est. 20-40 GB VRAM peak).
- **Regularization:** Dropout 0.1; early stopping on val CRPS (patience 10).

## 7. Performance Measurement

Blend metrics:
- **Distributional:** CRPS \( \int (F(z) - 1_{y \leq z})^2 dz \) (approx. via pinball); calibration error \( | (1/N) \sum 1_{y_i \leq \hat{q}_{\tau,i}} - \tau | \); PICP (e.g., 80% coverage for [0.1,0.9]); MPIW for sharpness.
- **Point:** IC (Spearman of median vs. actual); DA (sign accuracy); RMSE on median.
- **Financial:** Sharpe \( \sqrt{252 \times m} \cdot \bar{r} / \sigma_r \) (m= daily periods); MDD; profit factor.

This informs engineering by prioritizing testable concepts.

## Explanations for Disagreements with Feedback

- **Temporal Aggregation Causality:** Mild disagreement—strict causal masking in pooling is unnecessary here, as the lookback window is entirely historical (no future data), making bidirectional attention appropriate for capturing full context without risking lookahead. Clarified as bidirectional to preserve relative sequence importance (e.g., order of events).
- **RevIN Placement:** Agreed and explicitly placed in architecture flow (before embedding and reverse after decoder) for clarity.
- **Batch Size Range:** No disagreement—estimate is conservative but suitable for scientific ideation; exact profiling belongs in engineering.
- **Gradient Clipping:** Assess this as an implementation detail for the engineering document, not scientific ideation—it's a hyperparameter tuning choice rather than a core conceptual element.