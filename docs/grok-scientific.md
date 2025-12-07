# **grok-scientific.md**

## **1. Executive Scientific Mandate and Scope**

### **1.1 Directive and Role Definition**

This document, grok-scientific.md, establishes the scientific blueprint for ideating a Supervised Learning (SL) model to predict NASDAQ 100 E-mini futures (NQ) forward log-returns over horizons of 5m, 15m, 30m, 60m, 2h, and 4h, using ~1 trading week of 1-minute bars (6,825–6,900 bars) from June 2010 to December 2025 with volume-based rollover. As Grok, the chief quant/ML scientist, I synthesize insights from gemini-research.md, integrating and refining contributions from Claude and Gemini, including Claude's latest feedback on horizon weighting rationale and inference-time autoregressive behavior. This informs Claude's engineering plan.

Prioritizing accuracy/precision, the model employs Transformers with distributional outputs, adapting RL elements from MIGT (Instance Normalization, Lite Gate Units) and RL-TVDT (Variable Dependency Attention, Two-Stage Attention) to SL. We mandate full-sequence processing without patching to preserve temporal fidelity, avoiding foundation models for domain-specific optimization on A100 hardware.

### **1.2 Scientific Challenges and Hypotheses Overview**

NQ data features non-stationarity, multi-scale dependencies, and low SNR. Integrating team feedback, we hypothesize that full-context, gated, variable-aware attention will capture regime-invariant patterns, yielding IC > 0.05 and CRPS reductions of 25% over baselines, with explicit mitigations for computational and stability issues.

## **2. Ideation of Market Behavior**

### **2.1 NQ Microstructure and Behavioral Dynamics**

NQ exhibits volatility clustering, intra-day seasonality (e.g., open/close surges), and regime shifts (e.g., post-2008 recovery, COVID-19 volatility). Volume-based rollover mitigates discontinuities, but basis shifts require relative features. Hypothesis: Instance Normalization enables learning trajectory "shapes" invariant to price scales, improving cross-regime generalization by 15–20%.

Cyclical patterns (time-of-day volatility smiles, day-of-week anomalies like Friday expirations) suggest temporal priors enhance short-horizon precision. Multivariate interactions (price-volume-momentum) imply variable-centric modeling uncovers latent dependencies, testable via attention weights.

### **2.2 Noise, Outliers, and Signal Extraction**

Low SNR necessitates gating; LGU filters noisy updates, stabilizing gradients in volatile regimes. Distributional outputs capture heteroskedasticity and fat tails, hypothesizing superior tail accuracy (e.g., bimodal pre-event distributions) over point estimates.

## **3. Mathematical Formulations**

### **3.1 Input and State Space**

Let \( X \in \mathbb{R}^{T \times V} \) be the input, with \( T \approx 6900 \), \( V=24 \) features (23 derived + 1 gap). Features group semantically into 6 categories: price dynamics (\( \mathcal{F}_P \): log-returns \( r_t = \ln(C_t / C_{t-1}) \), high-low range, close location), volume (\( \mathcal{F}_V \): log-volume, delta, dollar volume), trend (\( \mathcal{F}_T \): VWAP deviation with reset at RTH open (09:30 ET) to capture session dynamics, MACD histogram, normalized SMA deviations), momentum (\( \mathcal{F}_M \): normalized RSI, CCI, DMI/ADX), volatility (\( \mathcal{F}_\sigma \): normalized ATR, Bollinger %B, bandwidth), and volume-weighted (\( \mathcal{F}_{VW} \): normalized MFI).

Group projection: For each group, concatenate raw features, linearly project to dim \( d \) (e.g., via \( W_g \in \mathbb{R}^{d \times |group|} \)), yielding 6 group embeddings; TSA operates on these \( V=6 \).

For variable \( T \): Set max \( T = 7000 \), right-pad with zeros, use key padding mask \( M \in \{0,1\}^T \) (1 for valid, 0 for pad; applies \( -\infty \) to padded attention scores). Positional embeddings encode actual timestamps; explicit gap feature \( \Delta t_i = \ln(1 + ((\text{timestamp}_i - \text{timestamp}_{i-1}) / 60 - 1)) \) as 24th feature (in a "temporal" group if needed), log-transformed for numerical stability.

### **3.2 Cyclical Positional Embeddings**

For cyclical \( c \) with period \( P \),
\[
\phi(c) = \left[ \sin\left(\frac{2\pi c}{P}\right), \cos\left(\frac{2\pi c}{P}\right) \right]
\]
Specifications: Time-of-day (\( P=1440 \)), day-of-month (\( P=31 \), fixed to avoid discontinuities at month boundaries), day-of-year (\( P=365.25 \)); day-of-week learnable \( E \in \mathbb{R}^{7 \times d} \). Composite:
\[
PE_t = W \cdot \text{Concat}[\phi(m_t), E(d^w_t), \phi(d^m_t), \phi(d^y_t)]
\]
Added to \( X_t \). Hypothesis: Captures seasonality, boosting intra-day IC by 0.03.

### **3.3 Instance Normalization**

\[
\text{IN}(x) = \gamma \frac{x - \mu(x)}{\sqrt{\sigma^2(x) + \epsilon}} + \beta
\]
Per sample, per feature across time, with learnable affine \( \gamma, \beta \) per variable. Statistics \( \mu, \sigma \) computed strictly on input window \( X_{1:T} \), excluding targets to prevent look-ahead bias. Hypothesis: Enforces stationarity, enabling regime-agnostic learning.

### **3.4 Two-Stage Attention (TSA)**

Input \( Z \in \mathbb{R}^{B \times T \times V \times d} \) (V=6 post-grouping, d=512).

Temporal (per variable): Reshape to \( (B \cdot V) \times T \times d \), apply MHA with mask \( M \) repeated across V:
\[
Z^{\text{time}}_{b,v} = \text{LN}(Z_{b,v} + \text{MHA}(Z_{b,v}, Z_{b,v}, Z_{b,v}; M))
\]
Variable (per time): Reshape to \( (B \cdot T) \times V \times d \), apply MHA (no mask needed on V dim):
\[
Z^{\text{var}}_{b,t} = \text{LN}(Z^{\text{time}}_{b,t} + \text{MHA}(Q_{b,t}, K_{b,t}, V_{b,t}))
\]
Full-sequence, no patching. Computational: With FlashAttention-2, memory scales linearly; based on benchmarks (e.g., seq=8k, B=32, d=2048 uses ~20-30GB), our params (seq=7k, B=8, d=512, V=6) estimated <20GB peak VRAM. Hypothesis: Decouples dimensions, enhancing interaction modeling (10% IC gain vs. vanilla).

### **3.5 Lite Gate Unit (LGU)**

Post-attention:
\[
z = \sigma(W_z x + U_z y), \quad h = \tanh(W_h x + U_h (z \odot y)), \quad o = (1-z) \odot x + z \odot h
\]
Initialize biases in \( W_z, U_z \) to favor \( z \approx 0.5 \) early, preventing gradient starvation. Hypothesis: Filters noise, reducing gradient variance by 40% in low-SNR periods without vanishing paths.

### **3.6 Distributional Output and Loss**

For each horizon \( h \), output quantiles \( \{\hat{y}_{\tau_k}\}_{k=1}^7 \) (\( \tau = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] \)). Pinball loss:
\[
\mathcal{L} = \sum_h \omega_h \sum_\tau \rho_\tau (y_h - \hat{y}_h^\tau), \quad \rho_\tau(u) = u(\tau - \mathbb{I}_{u<0})
\]
With \( \omega_h = 1 / \log(h + 1) \) to balance increasing variance over longer horizons (short horizons have lower variance; this upweights them to reflect higher precision, dampened by log to prevent dominance), plus monotonicity \( \mathcal{L}_{order} = \sum \max(0, \hat{y}^{\tau_j} - \hat{y}^{\tau_{j+1}}) \). Multi-horizon: Shared encoder, autoregressive heads where shorter horizons condition longer via residual connections (e.g., 5m output embeds into 15m head); use teacher forcing in training (feed ground truth short-horizon to longer heads). Hypothesis: Captures correlations and heteroskedasticity, improving tail accuracy by 20%.

## **4. Algorithmic Strategies**

### **4.1 Training Algorithm**

AdamW (lr=1e-4, warmup 2000 steps to peak, cosine decay), batch size 8–16, mixed precision. Steps: IN → Group projection + Embed + PE → TSA + LGU stacks → Extract last valid state \( H_T \) (post-mask) → Heads → Loss. Non-overlapping sampling for training, full for eval. Hypothesis: Warmup on low-vol data stabilizes convergence.

### **4.2 Inference Algorithm**

Rolling windows; ensemble via MC dropout. For autoregressive heads, feed predicted medians from shorter horizons into longer ones (e.g., 5m predicted median conditions 15m head) to maintain conditioning without ground truth. Threshold signals based on quantile spreads for risk-aware decisions.

## **5. Hypothesis Generation**

1. **Non-Stationarity (H1):** IN + PE yield 20% better cross-regime performance.
2. **Multi-Scale Dependency (H2):** Full-sequence TSA captures micro to weekly scales, improving long-horizon IC by 0.05.
3. **Variable Interaction (H3):** Variable attention uncovers correlations, testable by ablation (15% IC drop without).
4. **Heteroskedasticity (H4):** Quantile spreads predict variance, enabling dynamic sizing.
5. **Temporal Seasonality (H5):** Cyclical PE boosts precision in patterned periods (e.g., opens).
6. **No-Patching:** Full context preserves signals, outperforming patched baselines by 10% in short horizons.

Validate via ablations on holdouts (e.g., 2020–2025 OOS); phase gates: Proceed if IC > 0.03 on val, CRPS < baseline -10%.

## **6. Architecture Concept: "Grok-SL Transformer"**

Stack: IN → Group projection + Embed + PE → N blocks (TSA → LGU → FFN) → Extract last valid state \( H_T \) → Multi-horizon autoregressive quantile heads. N=8–12, d=256–512. Hypothesis: Balances fidelity and stability, yielding 25% CRPS improvement.

Extensions: Pre-Transformer convolutions for local patterns; explicit gap features.

## **7. Guidance for Engineering**

To Claude: Implement TSA with specified reshapes and group projection (concat + linear to d per group); benchmark VRAM in Phase 1; phase plan with hypothesis tests (Phase 1: Baseline Transformer with independent heads, val IC > 0.02; Phase 2: Add TSA/LGU/autoregressive with teacher forcing, CRPS drop >15%; Phase 3: Distributional heads, tail acc >55%).

## **8. Conclusion**

This refined document integrates team feedback, hypothesizing a full-context, gated Transformer for precise NQ predictions, grounded in gemini-research.md and extended for robustness.

## **9. References**

[1] F. Gu et al., "MIGT: Memory Instance Gated Transformer Framework for Financial Portfolio Management," arXiv preprint arXiv:2502.07280, 2025. [Online]. Available: https://arxiv.org/abs/2502.07280

[2] Y. Li et al., "Reinforcement Learning with Temporal and Variable Dependency-aware Transformer for Stock Trading Optimization," Neural Networks, vol. 192, p. 107905, 2025.

[3] A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems, 2017, pp. 5998–6008.

[4] W. Dabney et al., "Distributional Reinforcement Learning with Quantile Regression," in Proc. AAAI Conf. Artif. Intell., 2018, pp. 2892–2901.

[5] B. Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," Int. J. Forecasting, vol. 37, no. 4, pp. 1748–1764, 2021.

[6] "futuresSL_prob-statement.txt", User Uploaded Document.

[7] "gemini-research.md", Internal Project Document.