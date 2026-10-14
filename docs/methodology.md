# TerraPulse AI — Analytics Methodology

## 1. Trend Analysis: Seasonal Mann-Kendall & Sen's Slope
Standard linear regression on environmental time series violates independence assumptions due to seasonal autocorrelation and non-Gaussian sensor distributions.

1. **Seasonal Mann-Kendall (Hirsch et al. 1982)**:
   - Evaluates monotonic trend test statistics $S_p$ separately within each seasonal subseries $p \in [0, 22]$.
   - Test statistic:
     $$S_p = \sum_{i < j} \text{sign}(x_{j,p} - x_{i,p})$$
   - Aggregated variance $\text{Var}(S) = \sum \text{Var}(S_p)$ accounts for ties.
   - Evaluates test statistic $Z$ against the standard normal distribution.
2. **Sen's Slope Estimator (Theil-Sen)**:
   - Median of all pairwise slopes:
     $$Q = \text{median}\left(\frac{x_j - x_i}{j - i}\right), \quad \forall i < j$$
   - Insensitive to extreme outliers and seasonal spikes.

## 2. Leak-Free Climatological Anomaly Detection
Previous implementations that used centered rolling windows (`center=True`) suffer from look-ahead leakage, because values from $t+1, t+2$ enter the mean used to detect anomalies at $t$.
TerraPulse AI implements:
1. **Multi-Year Climatological Baseline**:
   - For composite period index $k \in [0, 22]$, historical mean $\mu_k$ and standard deviation $\sigma_k$ are derived across the multi-annual observation baseline.
   - Standardized anomaly score:
     $$Z_t = \frac{x_t - \mu_{k(t)}}{\sigma_{k(t)}}$$
   - Points with $|Z_t| \ge 2.0$ are classified as severe environmental anomalies.
2. **Strict Backward-Looking Rolling Baseline**:
   - Uses `shift=1` and `center=False` to ensure that time $t$ uses only $\{x_{t-1}, x_{t-2}, \dots\}$.

## 3. Bioclimatic Clustering (K-Means)
Unsupervised clustering of multi-annual joint NDVI and LST behavior:
- Features: Mean NDVI, LST, and phenological cycle index.
- Silhouette evaluation across candidate counts $K \in [2, 5]$.
- Optimal $K=2$ achieves Silhouette Score of $0.5142$:
  - **Cluster 0**: Moderate Canopy / Agro-Pastoral ($\text{NDVI} \approx 0.218$, $\text{LST} \approx 23.4°\text{C}$).
  - **Cluster 1**: Arid / High Thermal Stress ($\text{NDVI} \approx 0.151$, $\text{LST} \approx 38.7°\text{C}$).
