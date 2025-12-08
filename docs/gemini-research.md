# gemini-research.md: Research Guide for Distributional Supervised Learning on NASDAQ Futures

## 1. Executive Summary & Epistemological Framework

This document outlines the research foundations for constructing a state-of-the-art Supervised Learning (SL) model to predict NASDAQ 100 (NQ) futures returns. The core challenge is extracting signal from high-noise financial time series while explicitly modeling the *uncertainty* of those predictions.

To achieve the "accuracy/precision" priority mandated by the problem statement, we reject simple point-forecasting (Mean Squared Error) in favor of **Distributional Forecasting**. Furthermore, we move beyond standard Transformer architectures by integrating **Variable Embeddings** (from RL-TVDT) and **Gated Instance Normalization** (from MIGT) to handle the non-stationarity and multivariate dependencies unique to financial data.

This guide synthesizes the architectural insights of the *Scientific Lead* (Grok) and the engineering constraints identified by the *Engineering Lead* (Claude) into a unified research directive.

## 2. Data Ontology: Mechanics of NQ Futures

The model's efficacy depends entirely on the fidelity of its input data. The NQ futures market operates with specific microstructural rules that must be modeled explicitly.

### 2.1. Volume-Based Rollover and Back-Adjustment
*   **The Problem:** Futures contracts expire. Stitching them together creates price gaps. "Volume-based rollover" (switching to the next contract when its volume exceeds the current one) ensures we track liquidity, but the price jump between contracts introduces artificial volatility.
*   **The Solution:** The data pipeline must apply **Ratio Back-Adjustment**.
    *   *Method:* When rolling from Contract A to Contract B, calculate the ratio $R = \text{Price}_B / \text{Price}_A$. Multiply all historical data of Contract A by $R$.
    *   *Why:* Ratio adjustment preserves the percentage returns of the historical series, which is critical for log-return based machine learning models. Difference adjustment (adding/subtracting the spread) distorts percentage returns over long periods (2010–2025).

### 2.2. Handling Variable Sequence Lengths (273–276 Bars)
*   **The Constraint:** The 24-hour lookback window results in a variable number of bars (273 to 276) due to historical changes in trading halts and maintenance windows.
*   **The Solution:** **Padding with Attention Masks**.
    *   We define a fixed tensor size of $T_{max} = 288$ (theoretical max 5-min bars in 24h).
    *   Valid data is populated; invalid/missing timestamps are padded with zeros.
    *   A boolean **Attention Mask** is generated where `True` indicates a valid token and `False` is padding. This mask is passed to the Transformer's attention mechanism to prevent the model from "attending" to the padded zeros, ensuring the padding does not corrupt the feature extraction.[1]

## 3. Architectural Concept: The "MIGT-TVDT" Hybrid

Standard Transformers treat time series as a sequence of time-steps vectors $X_t \in \mathbb{R}^D$. Recent research (RL-TVDT, iTransformer) suggests this is suboptimal for multivariate series because it entangles temporal and variable correlations too early.

We propose a hybrid architecture that synthesizes **RL-TVDT** (for dependency modeling) and **MIGT** (for stability).

### 3.1. Variable Embedding (from RL-TVDT)
Instead of embedding a time-step, we embed the *entire time-series of a specific variable* into a token.
*   **Concept:** Input shape transforms from $(Batch, Time, Vars)$ $\to$ $(Batch, Vars, D_{model})$.
*   **Benefit:** This allows the attention mechanism to calculate the correlation between *RSI* and *Volume*, rather than just between *Time $t$* and *Time $t-1$*. This effectively learns a dynamic correlation graph between technical indicators [2],.[3]

### 3.2. Two-Stage Attention (TSA)
Proposed in RL-TVDT [2], this mechanism decouples the learning process:
1.  **Temporal Attention:** Applied independently to each variable's embedding. It learns the temporal dynamics (trends, seasonality) of *Price*, *Volume*, and *RSI* separately.
2.  **Variable Attention:** Applied across the variable tokens. It learns how *Volume* shocks impact *Price* movements.
*   **Mathematical Context:** This reduces the attention complexity and imposes a strong inductive bias that aligns with financial intuition (indicators drive price).

### 3.3. Gated Instance Normalization (from MIGT)
Financial data is non-stationary; the statistical properties of NQ in 2010 are different from 2025.
*   **Instance Normalization:** Normalizes inputs per-window (per instance), centering them to zero mean and unit variance. This allows the model to learn "shape" patterns (e.g., a "V" reversal) regardless of the absolute price level.[4]
*   **Lite Gate Unit (LGU):** A gating mechanism that filters the output of the attention layers. It allows the model to suppress noisy signals (common in 5-min bars) and pass through only high-confidence feature updates.[4]

## 4. Positional Encodings: The Geometry of Financial Time

The user requires specific embeddings for time. Standard sinusoidal encodings are insufficient for the multiple cycles of financial markets. We recommend a composite embedding approach:

1.  **Time of Day (Intraday):** **Cyclical Encoding**.
    *   $PE_{tod} = [\sin(2\pi t / 288), \cos(2\pi t / 288)]$. This ensures 23:55 is mathematically "close" to 00:00.
2.  **Day of Week (Weekly):** **Learnable Embedding**.
    *   Map Mon-Fri to a learned vector $E_{dow} \in \mathbb{R}^{d}$. Markets behave differently on "Option Expiry Fridays" vs "Mondays".
3.  **Day of Month/Year (Seasonal):** **Time2Vec**.
    *   Use **Time2Vec** [5], which learns a frequency $\omega$ and phase $\phi$ from the data: $\text{T2V}(\tau) = \sin(\omega \tau + \phi)$. This allows the model to discover non-obvious cycles (e.g., quarterly rebalancing flows) that fixed sinusoidal encodings might miss.

## 5. Derived Feature Engineering

The input vectors must capture higher-order properties of the market state.

*   **Volatility:** **Garman-Klass Volatility**. Uses Open, High, Low, and Close to estimate variance more efficiently than Close-to-Close returns. Critical for the "distributional" aspect of the model.[6]
*   **Liquidity/Microstructure:** **Amihud Illiquidity Proxy**. Calculated as $|R_t| / (P_t \times Vol_t)$. This captures "price impact"—how much price moves per unit of volume. High values indicate low liquidity/high fragility.[7]
*   **Momentum:** **RSI** and **MACD** (normalized).
*   **Trend:** **slope of EMAs** (derivatives of price action).

## 6. Distributional Output & Loss Functions

To predict *returns* with precision and model uncertainty, we must output a distribution, not a single number.

### 6.1. Quantile Regression (Recommended)
The model outputs a vector of quantiles $\tau \in \{0.1, 0.25, 0.5, 0.75, 0.9\}$ for each forecast horizon.
*   **Loss Function:** **Pinball Loss** (Quantile Loss).
    *   $L(y, \hat{y}_\tau) = \max(\tau(y - \hat{y}_\tau), (1-\tau)(\hat{y}_\tau - y))$.
*   **Why:** This directly optimizes the model to find the median (0.5) for accuracy, while the spread between the 0.1 and 0.9 quantiles provides a calibrated **Confidence Interval** (Precision). If the spread widens, the model is "uncertain," signaling risk.[8]

### 6.2. Categorical Distribution (Alternative - C51)
Discretizes returns into $N$ bins (e.g., -2% to +2%). The model predicts the probability mass for each bin using Softmax.
*   **Why:** Allows for multi-modal predictions (e.g., "market will likely go up big OR down big, but not stay flat").

## 7. Performance Measurement

Metrics must satisfy both ML and Quant perspectives.

1.  **Information Coefficient (IC):** Spearman correlation between predicted median return and actual return. Measures the *ranking* quality of predictions.
2.  **Sharpe Ratio:** Annualized mean return divided by std dev of returns. The gold standard for financial performance.
3.  **Continuous Ranked Probability Score (CRPS):** A strictly proper scoring rule that measures the accuracy of the *entire predicted distribution*, not just the mean. Crucial for distributional models.[9]
4.  **Tail-Weighted Calibration:** Measures how often actual returns fall outside the predicted 10th/90th percentiles.

## 8. Reference List

[2] Y. Li et al., "Reinforcement learning with temporal and variable dependency-aware transformer for stock trading optimization," *Neural Networks*, vol. 192, p. 107905, 2025. [Online]. Available: [https://arxiv.org/abs/2408.12446](https://arxiv.org/abs/2408.12446)
[4] F. Gu et al., "MIGT: Memory Instance Gated Transformer Framework for Financial Portfolio Management," *arXiv preprint arXiv:2502.07280*, 2025. [Online]. Available: [https://arxiv.org/abs/2502.07280](https://arxiv.org/abs/2502.07280)
[10] Y. Liu et al., "iTransformer: Inverted Transformers are Effective for Time Series Forecasting," *ICLR*, 2024. [Online]. Available: [https://arxiv.org/abs/2310.06625](https://arxiv.org/abs/2310.06625)
[1] A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017. [Online]. Available: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
[3] Y. Zhang and J. Yan, "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting," *ICLR*, 2023. [Online]. Available: [https://openreview.net/forum?id=vSVLM2j9eie](https://openreview.net/forum?id=vSVLM2j9eie)
[5] S. M. Kazemi et al., "Time2Vec: Learning a Vector Representation of Time," *arXiv preprint arXiv:1907.05321*, 2019. [Online]. Available: [https://arxiv.org/abs/1907.05321](https://arxiv.org/abs/1907.05321)
[6] M. B. Garman and M. J. Klass, "On the Estimation of Security Price Volatilities from Historical Data," *Journal of Business*, vol. 53, no. 1, pp. 67-78, 1980.
[7] Y. Amihud, "Illiquidity and stock returns: cross-section and time-series effects," *Journal of Financial Markets*, vol. 5, no. 1, pp. 31-56, 2002.
[8] R. Koenker and G. Bassett Jr, "Regression Quantiles," *Econometrica*, vol. 46, no. 1, pp. 33-50, 1978.
[9] T. Gneiting and A. E. Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation," *Journal of the American Statistical Association*, vol. 102, no. 477, pp. 359-378, 2007.