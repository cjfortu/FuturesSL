# **gemini-research.md**

## **1\. Executive Research Mandate and Scope**

### **1.1 Research Directive and Role Definition**

This document constitutes the definitive research guide for the gemini-research.md initiative, serving as the central reference point for the development of a Supervised Learning (SL) model targeting NASDAQ 100 E-mini futures (NQ). In accordance with the project hierarchy 1, this guide synthesizes market microstructure analysis, mathematical formulation, and architectural conception to direct the downstream scientific ideation (assigned to Grok) and engineering implementation (assigned to Claude). The objective is to define the search space for a model capable of ingesting high-frequency, long-context financial time series to predict forward returns over multiple horizons ranging from 5 minutes to 4 hours.1  
The formulation of this guide relies on a rigorous analysis of the provided primary literature—specifically the *Memory Instance Gated Transformer (MIGT)* 1 and *Reinforcement Learning with Temporal and Variable Dependency-aware Transformer (RL-TVDT)* 1 frameworks. While these papers operate within Reinforcement Learning (RL) paradigms, this research adapts their architectural innovations—Instance Normalization, Lite Gate Units, and Variable Dependency Attention—to a Supervised Learning context. This adaptation addresses the user's explicit requirement to utilize Transformers and distributional methods for return prediction, leveraging the A100 computational environment.1

### **1.2 The Problem of High-Dimensional Financial Prediction**

The core challenge is the prediction of NQ forward returns using a context window of approximately one trading week, translating to a sequence length of 6,825 to 6,900 one-minute bars.1 This creates a high-dimensional inputs space characterized by significant noise, non-stationarity, and complex inter-variable dependencies.  
Traditional econometrics (ARIMA, GARCH) fail to capture the non-linear interactions present in such high-dimensional data. Standard deep learning approaches (LSTMs, vanilla Transformers) often suffer from gradient instability or inability to model the distinct semantic roles of different market variables (e.g., price vs. volume vs. momentum). The research mandates a shift toward architectures that explicitly model these dependencies. The *RL-TVDT* framework 1 introduces the concept of separating temporal learning from variable learning, a critical insight for high-dimensional state spaces. Similarly, the *MIGT* framework 1 addresses the issue of signal-to-noise ratio through gating mechanisms. This guide integrates these concepts to propose a rigorous SL architecture.

## ---

**2\. Market Microstructure and Data Dynamics**

### **2.1 NASDAQ 100 Futures (NQ) Characteristics**

The NASDAQ 100 E-mini futures (NQ) represent a leverage-sensitive, high-beta instrument tracking the Nasdaq-100 index. The microstructure of NQ is defined by high volatility clustering and distinct intra-day seasonality. The research requires the use of "front month 1min bar data with volume-based rollover" from June 2010 to December 2025\.1

#### **2.1.1 Volume-Based Rollover and Price Continuity**

Futures contracts have expiration dates. To construct a continuous time series for long-term training, contracts must be stitched together. The "volume-based rollover" method specifies switching to the next contract month (e.g., H to M, M to U, U to Z) when the daily volume of the deferred contract exceeds that of the front month.  
The critical research implication involves price gaps. When rolling from one contract to the next, a price difference (the "roll gap") exists due to the basis (cost of carry, dividends, and interest rates). If unadjusted, these gaps introduce artificial jumps in the price series that the model might misinterpret as market shocks. However, the target variable is *forward returns*. Returns are relative measures ($p\_t / p\_{t-1}$).

* **Research Insight:** For a supervised learning model predicting returns, back-adjustment of absolute prices is preferred to maintain the integrity of technical indicators that depend on price levels (e.g., Moving Averages). However, if the indicators are calculated *before* the roll or reset at the roll, and the returns are calculated strictly within contracts, the impact is mitigated. Given the distributional nature of the target, log-returns $r\_t \= \\ln(P\_t) \- \\ln(P\_{t-1})$ provide the necessary stationarity and are robust to price level shifts, provided the calculation does not cross the rollover gap blindly.

### **2.2 Stationarity and Regime Shifts**

Financial time series are inherently non-stationary; their statistical properties (mean, variance) change over time. The requested period (2010–2025) encompasses multiple distinct market regimes: the post-2008 recovery, the low-volatility bull run of 2017, the COVID-19 crash and recovery (2020–2021), and the inflation-driven volatility of 2022\.  
The *MIGT* paper 1 explicitly addresses this via **Instance Normalization (IN)**. Unlike Layer Normalization (LN), which learns global affine parameters, IN normalizes each individual input sample (the 1-week window) to zero mean and unit variance. This effectively removes the absolute price level, allowing the model to focus on the *shape* of the price trajectory rather than its magnitude. This is crucial for a model trained on 2010 data (NQ \~2000) to generalize to 2025 data (NQ \~20,000).

### **2.3 Handling Variable Sequence Lengths**

The user queries: "How do we handle the variation in number of 1min bars in a trading week?".1  
A "trading week" is not a fixed unit of time due to holidays, early market closures, and the varying nature of trading sessions. A standard week might have \~6,900 minutes, but a holiday week significantly fewer.  
**Methodological Recommendation:**

1. **Fixed Context Window (Padding):** The Transformer architecture requires fixed tensor shapes for efficient batch processing. The sequence length $L$ should be set to the maximum expected bars (e.g., 7,000). Missing bars (due to market closures) are padded with a neutral token (e.g., zeros).  
2. **Attention Masking:** A "Key Padding Mask" must be generated for each sample. This boolean mask ensures the Self-Attention mechanism assigns $-\\infty$ scores to padded tokens, preventing them from influencing the probability distribution.  
3. **Temporal Integrity:** Unlike NLP, where "sentences" are discrete, financial time is continuous. If a holiday occurs, the time difference between Bar $t$ and Bar $t+1$ is not 1 minute but perhaps 24 hours \+ 1 minute. The positional embeddings (discussed in Section 3\) must encode this *timestamp* continuity, not just the index continuity. This ensures the model "sees" the gap.

## ---

**3\. Positional Embedding Strategy**

The user explicitly requests a strategy for encoding: *(1) time of day, (2) day of the week, (3) day of the month, and (4) day of the year*.1 In financial Transformers, the standard sinusoidal encoding used in NLP ($PE\_{(pos, 2i)} \= \\sin(pos/10000^{2i/d\_{model}})$) is suboptimal because "position" has semantic temporal meaning in markets (e.g., volatility is consistently higher at the 09:30 ET open).

### **3.1 Theory of Cyclical Temporal Encoding**

Time is a cyclical variable. The transition from 23:59 to 00:00 is seamless, yet a linear scalar representation (0 to 1439\) implies a maximal distance. To preserve the topological structure of time, we utilize trigonometric projection. For any cyclical feature $x\_t$ with period $P$, we project it onto a unit circle in 2D space:

$$\\phi\_{sin}(x\_t) \= \\sin\\left(\\frac{2\\pi x\_t}{P}\\right), \\quad \\phi\_{cos}(x\_t) \= \\cos\\left(\\frac{2\\pi x\_t}{P}\\right)$$  
This encoding ensures that the Euclidean distance between the encoding of the end of the cycle and the start of the cycle is minimal, mirroring reality.

### **3.2 Implementation Specifications for NQ Futures**

The research defines the following embedding specification to satisfy the user request:

| Feature | Period (P) | Encoding Method | Rationale |
| :---- | :---- | :---- | :---- |
| **Time of Day** | 1440 (min) | Continuous Trigonometric | Captures intra-day volatility smiles (Open/Close volume surges). |
| **Day of Week** | 7 (days) | Learnable Embedding | Captures distinct behavioral regimes (e.g., "Turnaround Tuesday", Friday expirations). |
| **Day of Month** | \~30.44 | Continuous Trigonometric | Captures monthly flows (e.g., 401k inflows, end-of-month rebalancing). |
| **Day of Year** | \~365.25 | Continuous Trigonometric | Captures seasonal trends (e.g., "Santa Claus Rally", "Sell in May"). |

Synthesis of Approaches:  
For "Day of Week," a Learnable Embedding (lookup table of size 7 $\\times$ $d\_{emb}$) is recommended over trigonometric encoding. Market behavior on Monday is structurally distinct from Friday due to weekend risk and option expirations, distinct from a purely smooth cycle. For "Time of Day," "Day of Month," and "Day of Year," the continuous trigonometric approach allows the model to interpolate and understand proximity effectively.  
Composite Positional Vector:  
The final positional embedding $PE\_t$ for time step $t$ is the concatenation of these features, projected to the model dimension $d\_{model}$:

$$PE\_t \= \\text{Linear}(\\text{Concat}\[\\phi(t\_{day}), E(d\_{week}), \\phi(d\_{month}), \\phi(d\_{year})\])$$

This vector is added to the feature embedding $X\_t$ before entering the Transformer encoder.

## ---

**4\. Feature Engineering and State Space Construction**

The user asks: "What derived features are optimal for this?".1  
The literature, particularly RL-TVDT 1 and MIGT 1, provides a robust, empirically validated set of features necessary for capturing the "Variable Dependency" in financial markets.

### **4.1 Insights from RL-TVDT: Variable-Centric Features**

The *RL-TVDT* paper emphasizes that market data is not just a sequence of scalar prices but a multivariate system where variables interact. It employs a "Variable Embedding" technique where features are treated as distinct entities. The recommended feature set includes:

1. **Trend Indicators:**  
   * **VWAP (Volume Weighted Average Price):** Critical for futures. It represents the liquidity-weighted fair value. The feature should be the deviation of Price from VWAP: $(P\_t \- \\text{VWAP}\_t) / P\_t$.  
   * **MACD (Moving Average Convergence Divergence):** Measures trend momentum. The histogram (difference between MACD line and Signal line) provides a stationary signal of trend acceleration.  
   * **SMA (Simple Moving Averages):** Multiple horizons (e.g., 20, 50, 200 bars). To ensure stationarity, these must be normalized relative to the current close price: $\\ln(P\_t / \\text{SMA}\_t)$.  
2. **Volatility Indicators:**  
   * **Bollinger Bands:** *RL-TVDT* explicitly utilizes these. The optimal derived feature is the **%B** indicator: $(P\_t \- \\text{LowerBand}\_t) / (\\text{UpperBand}\_t \- \\text{LowerBand}\_t)$, which indicates the position of the price relative to the volatility bands.  
   * **Bandwidth:** $(\\text{UpperBand}\_t \- \\text{LowerBand}\_t) / \\text{MiddleBand}\_t$. This serves as a direct proxy for volatility compression (the "squeeze"), often a precursor to expansion and breakouts.  
3. **Momentum Indicators:**  
   * **CCI (Commodity Channel Index):** Measures deviation from the statistical mean.  
   * **DMI (Directional Movement Index):** Deconstructs trend into Positive ($+DI$) and Negative ($-DI$) components, vital for the "Variable Attention" module to distinguish buying pressure from selling pressure.

### **4.2 Insights from MIGT: Signal Processing and Gating**

The *MIGT* framework 1 focuses on noise reduction. It recommends features that combine price and volume to confirm signal validity:

1. **MFI (Money Flow Index):** A volume-weighted RSI. For NQ futures, volume is the fuel of price movement. Divergences between Price and MFI are high-probability reversal signals.  
2. **RSI (Relative Strength Index):** The standard momentum oscillator. *MIGT* uses this to gauge overbought/oversold conditions, which informs the "Gating" mechanism (LGU) on when to attend to contrarian signals.  
3. **True Range (TR):** Measures the absolute volatility of a bar.

### **4.3 Recommended Feature Set for Supervised Learning**

Combining the research from both papers, the optimal feature vector $x\_t$ for each 1-minute bar should include:

| Feature Category | Derived Feature | Calculation / Transformation | Rationale |
| :---- | :---- | :---- | :---- |
| **Price Dynamics** | Log-Returns | $\\ln(P\_t / P\_{t-1})$ | Primary stationarity; input for distribution. |
| **Volume Dynamics** | Log-Volume Delta | $\\ln(V\_t / V\_{t-1})$ | Captures flow surges relative to recent history. |
| **Trend** | VWAP Deviation | $(P\_t \- \\text{VWAP}) / P\_t$ | Institutional benchmark relative value. |
| **Momentum** | RSI (14) | Standard RSI / 100 | Bounded momentum signal. |
| **Volatility** | Normalized ATR | $\\text{ATR}\_{14} / P\_t$ | Volatility scaled by price level. |
| **Statistical** | Z-Score (Price) | $(P\_t \- \\mu\_{window}) / \\sigma\_{window}$ | Explicit distributional position. |

Handling Stationarity:  
Crucially, raw prices ($P\_t$) must generally be excluded or strictly normalized (via Instance Normalization) as they are non-stationary. The model should learn from rates of change and relative positions, not absolute index values (e.g., NQ @ 15,000).

## ---

**5\. Theoretical Framework: The Transformer Architecture**

The user requires the use of Transformers. The *MIGT* 1 and *RL-TVDT* 1 papers provide specific architectural modifications to the standard Transformer that are essential for financial time series. We must adapt these from their native RL context to the user's Supervised Learning (SL) problem.

### **5.1 The RL-TVDT Architecture: Variable Dependency Attention**

The *RL-TVDT* paper identifies a critical weakness in standard time-series Transformers: they often embed all variables at a time step $t$ into a single vector $h\_t$, losing the distinct information of individual features (e.g., confusing a drop in RSI with a drop in Price).

#### **5.1.1 Variable Embedding Strategy**

Instead of a single vector $x\_t \\in \\mathbb{R}^{D\_{model}}$ representing time $t$, *RL-TVDT* proposes representing the input as a tensor $X \\in \\mathbb{R}^{T \\times V \\times D}$, where $T$ is time steps, $V$ is the number of variables (features), and $D$ is the embedding dimension.

* **Segmentation:** To handle the computational load, *RL-TVDT* segments the time series of each variable into patches (e.g., length $L\_{seg} \= 10$). Each segment is projected linearly.  
* **Implication for NQ Model:** With a 6,900-bar context, segmentation is not just an optimization; it is a necessity. Using a patch size of $P=15$ (15 minutes) reduces the sequence length from 6,900 to 460\. This allows the model to attend to the global context (the week) without exploding memory usage.

#### **5.1.2 Two-Stage Attention (TSA) Mechanism**

*RL-TVDT* introduces a TSA block that processes the data in two orthogonal dimensions:

1. Stage 1: Temporal Attention. The model attends across time for each variable independently. It asks: "How does the RSI sequence relate to past RSI values?"

   $$Z^{time} \= \\text{LayerNorm}(Z \+ \\text{MHA}^{time}(Z, Z, Z))$$  
2. Stage 2: Variable Attention. The model attends across variables at each time step. It asks: "Given the current RSI and MACD, what is the joint market state?"

   $$Z^{var} \= \\text{LayerNorm}(Z^{time} \+ \\text{MHA}^{var}(Q\_{var}, K\_{var}, V\_{var}))$$  
   * *Research Insight:* This effectively models the interaction graph of technical indicators. For the SL model, this architecture is superior to "Vanilla" Transformers which mash all features together immediately.

### **5.2 The MIGT Architecture: Gating and Stability**

The *MIGT* paper addresses the "unstable training" and "outlier impact" inherent in financial data.

#### **5.2.1 Instance Normalization (IN)**

As discussed in Section 2.2, MIGT applies IN to the input window.

$$\\text{IN}(x) \= \\frac{x \- \\mu(x)}{\\sigma(x) \+ \\epsilon}$$

where $\\mu(x)$ and $\\sigma(x)$ are calculated across the time dimension of the specific sample. This ensures the Transformer sees a standardized distribution regardless of whether the market is at 10,000 or 20,000 points.

#### **5.2.2 Lite Gate Unit (LGU)**

Financial data is low SNR (Signal-to-Noise Ratio). A standard Transformer's residual connection ($x \+ \\text{Attention}(x)$) forces the model to incorporate the attention output even if it is noise. *MIGT* introduces the LGU to selectively filter updates:

$$\\begin{aligned} z &= \\sigma(W\_z x \+ U\_z y) \\\\ h &= \\text{tanh}(W\_g x \+ U\_g (z \\odot y)) \\\\ \\text{Output} &= (1-z) \\odot x \+ z \\odot h \\end{aligned}$$  
where $y$ is the output of the attention mechanism.

* **Application:** This gating mechanism should be integrated into the Feed-Forward Network (FFN) sub-layers of the user's SL Transformer. It allows the model to effectively "skip" processing for noisy time segments, preserving the clean signal from the residual path.

### **5.3 Synthesis: The "Gemini-SL" Architecture Concept**

Integrating these findings, the proposed architecture for the NQ model is:

1. **Input:** $T \\approx 6900$, $V$ features.  
2. **Preprocessing:** Instance Normalization per variable.  
3. **Patching:** Segment into non-overlapping patches of size $P$ (e.g., 5 or 15 mins).  
4. **Embedding:** Linear projection of patches \+ Cyclical Positional Embeddings.  
5. **Encoder:** Stack of $N$ blocks containing:  
   * **TSA (Two-Stage Attention):** First Temporal, then Variable.  
   * **LGU (Lite Gate Unit):** Gating the output of the attention blocks.  
6. **Head:** Distributional Projection (discussed in Section 6).

## ---

**6\. Mathematical Reasoning: Distributional Learning**

The user specifies: "The model will use... distributional methods.".1  
Standard regression minimizes Mean Squared Error (MSE), which assumes a fixed-variance Gaussian distribution of errors ($y \\sim \\mathcal{N}(\\mu, \\sigma^2\_{const})$). Financial returns are leptokurtic (fat-tailed) and heteroskedastic (variance changes over time). Minimizing MSE leads to underestimating tail risk, which is catastrophic for futures trading.

### **6.1 Quantile Regression (Non-Parametric)**

Instead of predicting a single value $\\hat{y}$, the model predicts a set of conditional quantiles $\\{\\hat{y}\_{\\tau\_1}, \\hat{y}\_{\\tau\_2}, \\dots, \\hat{y}\_{\\tau\_k}\\}$ (e.g., 10th, 50th, 90th percentiles).  
The objective function is the Pinball Loss (or Tiled Quantile Loss):

$$\\mathcal{L}\_{QR} \= \\sum\_{i=1}^{k} \\mathbb{E}\_{(x,y) \\sim \\mathcal{D}} \[\\rho\_{\\tau\_i}(y \- \\hat{y}\_{\\tau\_i}(x))\]$$  
where the check function $\\rho\_\\tau(u)$ is defined as:

$$\\rho\_\\tau(u) \= u(\\tau \- \\mathbb{I}(u \< 0)) \= \\max(\\tau u, (\\tau-1)u)$$  
**Why Optimal for NQ:**

* **Robustness:** The median ($\\tau=0.5$) is robust to outliers, unlike the mean (MSE).  
* **Risk Estimation:** The distance between the 90th and 10th percentiles ($\\hat{y}\_{0.9} \- \\hat{y}\_{0.1}$) provides a dynamic estimate of prediction uncertainty (volatility). If this interval widens, the model is signaling "high risk/low confidence."

### **6.2 Gaussian Negative Log-Likelihood (Parametric)**

The model outputs two scalars: mean $\\mu(x)$ and log-variance $s(x) \= \\log(\\sigma^2(x))$. The loss is the Negative Log-Likelihood (NLL) of the Gaussian:

$$\\mathcal{L}\_{NLL} \= \\frac{1}{2} \\log(2\\pi) \+ \\frac{1}{2}s(x) \+ \\frac{(y \- \\mu(x))^2}{2 e^{s(x)}}$$

* **Heteroskedasticity:** The term $e^{s(x)}$ in the denominator acts as a learned attenuation mechanism. If the model is uncertain (high $s(x)$), the penalty for the error $(y-\\mu)^2$ is reduced. This allows the model to "survive" high-volatility periods without destroying its gradients, while learning to predict the volatility itself.

### **6.3 Categorical Distribution (C51 / Classification)**

Inspired by Distributional RL (C51), we can discretize the return space into $N$ bins (atoms) ranging from $V\_{min}$ (e.g., \-2%) to $V\_{max}$ (+2%). The model outputs a softmax probability vector $p$ over these bins.

* **Loss:** Cross-Entropy between the predicted distribution and the projected true return (one-hot or smoothed).  
* **Benefit:** This can model multi-modal distributions. For example, before a major news event (CPI release), the distribution might be bimodal (large move up OR large move down, low probability of zero). Gaussian or Quantile models struggle to capture this bimodality explicitly.

Recommendation:  
Given the user's priority on "accuracy/precision," Quantile Regression is the most robust starting point. It makes no assumptions about the underlying distribution shape (unlike Gaussian) and does not require careful bin tuning (unlike Categorical). It directly provides the confidence intervals needed for sizing positions.

## ---

**7\. Performance Evaluation Methodology**

The user asks: "How do we measure model performance in a way meaningful to researchers in both AI/ML and quant/finance?".1  
Evaluating probabilistic forecasts requires metrics that assess both calibration (reliability) and sharpness (precision).

### **7.1 AI/ML Metrics (Probabilistic Rigor)**

1. Continuous Ranked Probability Score (CRPS): The gold standard for distributional accuracy. It generalizes Absolute Error to probabilistic forecasts. It measures the integral of the squared difference between the predicted Cumulative Distribution Function (CDF) $F$ and the empirical CDF of the observation $y$:

   $$\\text{CRPS}(F, y) \= \\int\_{-\\infty}^{\\infty} (F(z) \- \\mathbb{I}\\{y \\le z\\})^2 dz$$  
   * *Interpretation:* A lower CRPS means the predicted distribution is concentrated tightly around the true value.  
2. Negative Log-Likelihood (NLL): Measures how "surprising" the true data point is given the predicted distribution.

   $$\\text{NLL} \= \-\\log p(y | x)$$

### **7.2 Quant/Finance Metrics (Economic Utility)**

1. **Information Coefficient (IC):** The Pearson (or Spearman) correlation between the model's predicted expected return (e.g., the median $\\hat{y}\_{0.5}$) and the realized return $y\_{target}$.  
   * *Target:* IC \> 0.05 is generally considered significant in high-frequency futures.  
2. Sharpe Ratio (Backtest Proxy): While this is an SL model, we can derive a proxy metric.

   $$\\text{Signal}\_t \= \\text{sign}(\\hat{y}\_{0.5}) \\times \\mathbb{I}(|\\hat{y}\_{0.5}| \> \\text{cost})$$

   Calculate the annualized mean/std of the returns generated by this naive signal. The MIGT paper 1 uses Sharpe, Sortino, and Omega ratios. The Omega Ratio is particularly relevant as it considers higher moments (skewness/kurtosis), aligning with the distributional nature of the model.  
3. **Tail Weighted Accuracy:** Accuracy computed only on the top/bottom 10% of realized returns.  
   * *Rationale:* In trading, performance is dominated by the ability to capture (or avoid) the tails. A model with 60% accuracy on flat days but 0% on volatile days is useless.

## ---

**8\. Implementation and Computational Strategy**

### **8.1 Hardware Constraints and Optimization**

The environment is Google Colab A100 (80GB VRAM, 12 CPU cores).1  
Processing a sequence length of $L=6900$ with a standard Transformer (Attention complexity $O(L^2)$) generates an attention matrix of size $6900 \\times 6900 \\approx 47.6 \\times 10^6$ elements. With $H$ heads and $B$ batch size, this fits in 80GB VRAM but is inefficient.  
**Optimization Directives:**

1. **FlashAttention:** The implementation must utilize **FlashAttention-2**. This kernel fuses the attention operations, avoiding materialization of the full $N \\times N$ matrix in HBM (High Bandwidth Memory). This provides a speedup of 2-4x and linear memory scaling.  
2. **Mixed Precision (AMP):** Training should utilize **TF32** (TensorFloat-32) on the A100 for matrix multiplications and **BF16** (BFloat16) for storage/transport. This maintains numerical stability (unlike FP16) while doubling throughput.  
3. **DataLoader Strategy:** The bottleneck will likely be the CPU (12 cores) feeding the GPU.  
   * **Pre-computation:** Derived features (RSI, VWAP) should be pre-calculated and stored (e.g., in .parquet or .npy format), not computed on-the-fly during training.  
   * **Contiguous Memory:** Ensure the rolling window batches are stored contiguously in memory to maximize data transfer rates.

### **8.2 Scientific & Engineering Workflows**

To satisfy the user's workflow requirements 1, this research guide dictates the following next steps:

* **For Grok (Scientific Ideation):**  
  * Formulate the exact equations for the **Two-Stage Attention** combining Temporal and Variable dimensions.  
  * Derive the gradients for the **Pinball Loss** combined with the **LGU** gating mechanism to ensure the gating does not vanish the gradients for the distributional head.  
  * Design the specific **Variable Embedding** topology: Which features are grouped? Do we embed "Price" and "Volume" together or separately?  
* **For Claude (Engineering Implementation):**  
  * Design the Dataset class to handle the **Volume-Based Rollover** logic seamlessly, ensuring no look-ahead bias in the normalization.  
  * Implement the **Instance Normalization** layer as a custom nn.Module that can handle the 3D tensor structure (Batch, Time, Variable) correctly.  
  * Prototype the **FlashAttention** integration using PyTorch 2.0+ scaled\_dot\_product\_attention.

## ---

**9\. Conclusion**

This research guide establishes a rigorous foundation for the development of a Transformer-based Distributional Supervised Learning model for NQ futures. By synthesizing the architectural stability of **MIGT** (Instance Normalization, Gating) with the structural awareness of **RL-TVDT** (Variable Embeddings, Two-Stage Attention), and grounding these in robust **Distributional Learning** theory (Quantile Regression), the proposed framework directly addresses the challenges of non-stationarity, noise, and complex dependencies inherent in high-frequency financial data.  
The path forward requires Grok to mathematically formalize the "Gemini-SL" architecture and Claude to engineer the high-performance pipeline on the A100 infrastructure. The rigorous adherence to these principles will maximize the probability of achieving high-precision, risk-aware forward return predictions.

## **10\. References**

1 "project\_instructions\_trunc.txt", User Uploaded Document.  
1 "futuresSL\_prob-statement.txt", User Uploaded Document.  
1 F. Gu, A. Stefanidis, A. García-Fernández, J. Su, and H. Li, "MIGT: Memory Instance Gated Transformer Framework for Financial Portfolio Management," arXiv preprint arXiv:2502.07280, Feb. 2025\.  
1 Y. Li, X. Dong, Z. Wu, J. Gao, T. Zhang, and L. Yu, "Reinforcement Learning with Temporal and Variable Dependency-aware Transformer for Stock Trading Optimization," Neural Networks, vol. 192, p. 107905, 2025\.  
A. Vaswani et al., "Attention is all you need," in Advances in neural information processing systems, 2017\.  
B. Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," International Journal of Forecasting, 2021\.  
W. Dabney et al., "Distributional Reinforcement Learning with Quantile Regression," AAAI, 2018\.  
T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism," 2023\.  
D. Ulyanov et al., "Instance Normalization: The Missing Ingredient for Fast Stylization," 2016\.

#### **Works cited**

1. project\_instructions\_trunc.txt