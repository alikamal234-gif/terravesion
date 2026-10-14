# PREPROCESSING REPORT: NDVI & LST

## STATUS
**SUCCESS: ANALYSIS-READY DATASET PRODUCED AND VALIDATED**

## 1. Overview & Provenance
- **Output File**: `data/processed/terra_pulse_morocco_clean.nc`
- **File Format**: NETCDF4 / HDF5 (deflate/zlib compression level 4)
- **File Size**: 1008.81 MB (~1.01 GB)
- **Source NDVI**: NASA LP DAAC MOD13Q1.061 (nominal 250 m, 16-day composite)
- **Source LST**: NASA LP DAAC MOD11A2.061 (nominal 1 km, 8-day composite)
- **Boundary Dataset**: GeoBoundaries (gbOpen) Morocco ADM0 polygon (`data/raw/morocco_boundary.geojson`)
- **Target Spatial Grid**: ~1 km (exact pixel resolution: 926.625 m × 926.625 m)
- **Projection**: MODIS Sinusoidal (matching source data, avoiding reprojection distortions)
- **Target Temporal Resolution**: 16-day frequency matching actual NDVI timestamps (271 observations)
- **Temporal Span**: 2014-12-19 to 2026-09-14 (~11.75 years of continuous observations)

---

## 2. Operations Performed

### A. Raw Data Handling & Immutability
- Both raw datasets were accessed in strict read-only mode using `mask_and_scale=False` and `decode_times=False`.
- Verified that all raw files in `data/raw/ndvi/` and `data/raw/lst/` remained 100% unmodified (checksum and timestamp verified).

### B. Quality Filtering & Masking
1. **NDVI**:
   - Official `_FillValue = -3000` masked.
   - Raw valid integer range `[-2000, 10000]` preserved and scaled by `0.0001`, yielding valid physical NDVI range `[-0.20, 1.00]`.
2. **LST**:
   - Official `_FillValue = 0` masked.
   - Undocumented `65535` error/dense-cloud flag masked.
   - `QC_Day` Mandatory QA bitmask applied: `(QC_Day & 0b11) <= 1` (retains 'LST produced, good quality' and 'LST produced, other quality'; excludes cloud-obscured and failed retrievals).
   - Valid integer observations scaled by `0.02` to physical temperature in Kelvin.

### C. Spatial Alignment Method
- **Grid Geometry**: We confirmed that while the pixel resolution ratio between LST and NDVI is exactly $4.0$ ($926.6254\text{ m} / 231.6564\text{ m} = 4.0$), the northern boundary of NDVI has a 1-pixel ($231.66\text{ m}$) offset relative to LST.
- **Resampling**: Area-weighted spatial aggregation (`Resampling.average`) via `rioxarray` was used for each time slice, properly averaging the 16 underlying 250 m sub-pixels into each 1 km cell.

### D. Morocco Boundary Masking & Cropping
- The official Morocco ADM0 polygon was reprojected to MODIS Sinusoidal coordinates.
- Using `rasterio.features.geometry_mask`, a 2D boolean mask was rasterized onto the 1 km target grid.
- **795,887** pixels fall within mainland Morocco (24.44% of the rectangular bounding box).
- All pixels outside Morocco (ocean and international borders) were strictly set to `NaN`.

### E. Temporal Alignment Method
- The target temporal coordinates are the **actual 271 NDVI composite timestamps**.
- For each NDVI observation $[T_i, T_{i+1})$, all valid LST 8-day observations within the interval were aggregated using the arithmetic mean (`np.nanmean`).
- Missing observations (such as the 1 missing LST step in 2016) were handled without synthetic interpolation: if only 1 valid observation existed in the window, it represented the period.
- For 5 time periods where no LST data was captured (at the tail of the time series where LST composite delivery ended prior to NDVI), LST was preserved as `NaN`.

---

## 3. Validation Results

| Parameter | NDVI Variable | LST Variable |
|---|---|---|
| **Data Type** | `float32` | `float32` |
| **Grid Dimensions** | Time: 271, Y: 1807, X: 1802 | Time: 271, Y: 1807, X: 1802 |
| **Total Grid Cells** | 882,434,000 | 882,434,000 |
| **Minimum Value** | -0.1871 | 252.58 K (-20.57 °C) |
| **Maximum Value** | 0.9503 | 336.01 K (62.86 °C) |
| **Mean Value** | 0.1615 | 305.88 K (32.73 °C) |
| **NaN Percentage** | 76.10% | 76.43% |
| **Physical Sanity** | **PASSED** (valid vegetation & water envelope) | **PASSED** (realistic Moroccan temperature envelope) |
| **Raw Data Immutability** | **VERIFIED (100% Unmodified)** | **VERIFIED (100% Unmodified)** |

*Note on NaN Percentage: 75.56% of the bounding box consists of Atlantic/Mediterranean waters and territory outside Morocco, which were intentionally masked to NaN by the Morocco national boundary polygon.*

---

## 4. Scientific Limitations
1. **LST Data Gap**: LST contains 5 tail periods with missing data towards late 2026 where AppEEARS LST composite delivery ended prior to NDVI. These are strictly marked as NaN without synthetic imputation.
2. **Aggregation Smoothing**: Spatial aggregation from 250 m to ~1 km smooths out localized vegetation micro-patches, but ensures physical consistency with thermal infrared observations.
