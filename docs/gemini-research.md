# gemini-research.md: Research Guide for Hierarchical Distributional Supervised Learning on NASDAQ Futures

## 1. Executive Summary & Epistemological Framework

This document outlines the research foundations for constructing a state-of-the-art **Hierarchical Supervised Learning (SL)** model to predict NASDAQ 100 (NQ) futures returns. The core challenge is extracting signal from high-noise financial time series while explicitly modeling the *uncertainty* of those predictions across multiple temporal resolutions.

To achieve the "accuracy/precision" priority mandated by the problem statement, we move beyond simple point-forecasting and uniform time windows. We adopt a **"Sparse-Dense-Sparse" (SDS)** hierarchical approach, fusing high-frequency microstructure data with lower-frequency macro context. This allows the model to "remember" the market state from exactly one week prior while maintaining the granular precision required for a new **5-minute prediction horizon**.

This guide synthesizes the architectural insights of the *Scientific Lead* (Grok), the engineering specifications of the *Engineering Lead* (Claude), and the research analysis of the *Research Expert* (Gemini).

## 2. Data Ontology: The "SDS" Hierarchical Structure

The model's efficacy depends on feeding it a context window that mirrors how human traders analyze structure: immediate price action combined with higher-level trend and historical seasonality. We reject the flat 24-hour window in favor of a **Multi-Resolution Context Window**.

### 2.1. The Sparse-Dense-Sparse (SDS) Window
We define the input sequence $X_{input}$ as a concatenation of three distinct temporal segments:

1.  **Segment A: Weekly Echo (High-Res)**
    * **Time:** $T_{now} - 1\text{w}$ to $T_{now} - 1\text{w} + 1\text{h}$
    * **Resolution:** **1-minute bars**.
    * **Purpose:** Captures the "ghost" of the market—specific microstructural behaviors (e.g., liquidity gaps, algorithmic triggers) that occurred at this exact time last week. This addresses the "Weekly Seasonality" hypothesis.
2.  **Segment B: Interim Macro (Low-Res)**
    * **Time:** $T_{now} - 1\text{w} + 1\text{h}$ to $T_{now} - 4\text{h}$
    * **Resolution:** **60-minute bars**.
    * **Purpose:** Compresses the long "bridge" of history. It captures regime shifts, macro trend evolution, and volatility expansion/contraction without wasting compute on noise.
3.  **Segment C: Recent Micro (High-Res)**
    * **Time:** $T_{now} - 4\text{h}$ to $T_{now}$
    * **Resolution:** **1-minute bars**.
    * **Purpose:** Provides the immediate order flow, momentum, and volatility state required for the **5-minute prediction horizon**.

### 2.2. Multi-Scale Variable Embedding
Standard TVDT embeds variables independently. However, a "Close Price" on a 1-minute bar has different statistical properties (noise, variance) than a "Close Price" on a 60-minute bar.
* **The Solution:** Use **Distinct Projection Layers**.
    * $E_{micro}(x)$ projects variables from Segments A and C.
    * $E_{macro}(x)$ projects variables from Segment B.
    * This allows the model to learn scale-specific representations (e.g., a 10-point move in 1 minute is a shock; in 60 minutes, it is noise).

## 3. Architectural Concept: Hierarchical MIGT-TVDT

We propose a hybrid architecture that integrates **MIGT** (gating), **TVDT** (variable attention), and **Global Flash Attention** to handle the SDS structure.

### 3.1. Hybrid Patching Strategy
To optimize for the A100 GPU (80GB VRAM) without destroying the short-term signal:
* **Segment B (Macro):** Apply **Patching** (e.g., Patch Size 4 or 16). This aggregates hourly steps into denser tokens, reducing sequence length significantly.
* **Segments A & C (Micro):** **NO PATCHING** (Patch Size 1). We must preserve the raw 1-minute resolution to ensure the 5-minute prediction horizon is not subverted or smoothed out.

### 3.2. Global Flash Attention
We explicitly reject SWIN Transformers or local-window attention for this task.
* **Why:** The A100's 80GB VRAM allows us to process the entire concatenated sequence (Echo + Interim + Recent) using **Flash Attention 2**.
* **Benefit:** This enables "infinite" receptive fields where the *Recent Micro* tokens can directly attend to the *Weekly Echo* tokens. The model can instantly correlate a current price formation with a similar setup from last week without passing through intermediate layers.

### 3.3. The 5-Minute Horizon Head
To handle the new, ultra-short 5-minute horizon:
* **Dedicated Head:** We introduce a specific prediction head that reads *only* from the unpatched **Segment C (Recent)** embeddings.
* **Auxiliary Loss:** This head may carry a higher loss weight to force the encoder to retain high-frequency details, ensuring the hierarchical compression does not "wash out" the immediate signal.

## 4. Positional Encodings: Bridging the Gap

Relative positional encodings (standard in Transformers) fail here because the sequence has massive time gaps (the jump from "1 week ago" to "now") and varying speeds (1m vs 60m).

* **Absolute Time Encoding (Time2Vec):** We must use **Time2Vec** or timestamp-based embeddings to encode the *absolute* time distance.
* **Function:** This explicitly signals to the model that Segment A is exactly $T - 168\text{h}$ and Segment C is $T - 0\text{h}$, allowing it to compute the precise phase shift of the weekly cycle.

## 5. Derived Feature Engineering (Multi-Resolution)

Input vectors must now be computed consistent with their segment's resolution:
* **Micro Features (Segments A & C):** Calculated on 1-minute data (e.g., `RSI_14` represents 14 minutes).
* **Macro Features (Segment B):** Calculated on 60-minute data (e.g., `RSI_14` represents 14 hours).
* **Harmonization:** All features are normalized (RevIN) to ensure numerical stability across resolutions, but their *informational content* remains scale-specific.

## 6. Distributional Output & Loss Functions

We maintain the **Distributional Forecasting** approach.

### 6.1. Quantile Regression
The model outputs a vector of quantiles $\tau \in \{0.1, 0.25, 0.5, 0.75, 0.9\}$ for all horizons, including the new **5-minute horizon**.
* **Loss Function:** **Pinball Loss**.
* **Calibration:** We monitor the spread between $\tau_{0.1}$ and $\tau_{0.9}$ as a dynamic measure of market uncertainty.

## 7. Performance Measurement

Metrics must now account for the hierarchical nature of the prediction.

1.  **Horizon-Specific IC:** Calculate Information Coefficient (IC) separately for the 5-minute horizon vs. the 4-hour horizon. We expect the 5-minute horizon to be driven by microstructure (Segment C) and the 4-hour by trend (Segment B).
2.  **Weekly Seasonality Score:** A custom metric measuring the correlation between prediction accuracy and the "Weekly Echo" input, validating if the model is successfully utilizing Segment A.
3.  **Sharpe Ratio & CRPS:** Standard financial and probabilistic metrics remain the gold standard.

## 8. Reference List

[1] A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
[2] Y. Li et al., "Reinforcement learning with temporal and variable dependency-aware transformer for stock trading optimization," *Neural Networks*, vol. 192, p. 107905, 2025.
[3] Y. Zhang and J. Yan, "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting," *ICLR*, 2023.
[4] F. Gu et al., "MIGT: Memory Instance Gated Transformer Framework for Financial Portfolio Management," *arXiv preprint arXiv:2502.07280*, 2025.
[5] S. M. Kazemi et al., "Time2Vec: Learning a Vector Representation of Time," *arXiv preprint arXiv:1907.05321*, 2019.
[6] M. B. Garman and M. J. Klass, "On the Estimation of Security Price Volatilities from Historical Data," *Journal of Business*, vol. 53, no. 1, pp. 67-78, 1980.
[7] Y. Amihud, "Illiquidity and stock returns: cross-section and time-series effects," *Journal of Financial Markets*, vol. 5, no. 1, pp. 31-56, 2002.
[8] R. Koenker and G. Bassett Jr, "Regression Quantiles," *Econometrica*, vol. 46, no. 1, pp. 33-50, 1978.
[9] T. Gneiting and A. E. Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation," *Journal of the American Statistical Association*, vol. 102, no. 477, pp. 359-378, 2007.
[10] T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," *NeurIPS*, 2022.