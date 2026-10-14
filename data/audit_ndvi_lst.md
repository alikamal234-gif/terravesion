# RIGOROUS DATA SUFFICIENCY AUDIT: NDVI & LST

## 1. Exact Temporal Coverage

### NDVI (MOD13Q1.061)
- **First Timestamp**: 472262400.0 (seconds since 2000-01-01) -> ~2014-12-19
- **Last Timestamp**: 842659200.0 (seconds since 2000-01-01) -> ~2026-09-14
- **Time Units**: `seconds since 2000-01-01 00:00:00+00:00`
- **Calendar**: `julian`
- **Count**: 271 observations
- **Temporal Consistency**: **Perfectly Regular**. Expected step is exactly 16 days (1,382,400 seconds). No missing steps detected.

### LST (MOD11A2.061)
- **First Timestamp**: 472953600.0 (seconds since 2000-01-01) -> ~2014-12-27
- **Last Timestamp**: 836438400.0 (seconds since 2000-01-01) -> ~2026-07-04
- **Time Units**: `seconds since 2000-01-01 00:00:00+00:00`
- **Calendar**: `julian`
- **Count**: 530 observations
- **Temporal Consistency**: **Irregular**. Expected step is 8 days (691,200 seconds). **There is 1 missing time step gap**. This gap is likely a known Terra satellite outage or a missing composite in the LP DAAC archive.

---

## 2. Exact Spatial Coverage

### NDVI (MOD13Q1.061)
- **X Min / Max**: -1778078.38 / -108994.32
- **Y Min / Max**: 2320617.57 / 3993871.44
- **Coordinate Variables**: `xdim`, `ydim`
- **Resolution**: 231.656m (X and Y)
- **Spatial Reference**: Sinusoidal Projection (MODIS standard)

### LST (MOD11A2.061)
- **X Min / Max**: -1777730.89 / -108878.49
- **Y Min / Max**: 2320733.40 / 3994218.93
- **Coordinate Variables**: `xdim`, `ydim`
- **Resolution**: 926.625m (X and Y)
- **Spatial Reference**: Sinusoidal Projection (MODIS standard)

*Both bounding boxes cover the identical projected area covering Morocco. However, the exact edge coordinates differ slightly because they snap to their respective grid resolutions (250m vs 1km).*

---

## 3. Global Attributes
Both datasets share the following global attributes inherited from NASA AppEEARS:
- **Conventions**: CF-1.6
- **Institution**: Land Processes Distributed Active Archive Center (LP DAAC)
- **Source**: AppEEARS v3.131
- **File Format**: NETCDF4 / HDF5

---

## 4 & 5. NDVI Quality Audit
- **Main Variable**: `_250m_16_days_NDVI`
- **Scale Factor**: 0.0001
- **Add Offset**: 0.0
- **_FillValue**: -3000
- **Valid Range**: -2000 to 10000 (Physical NDVI range: -0.2 to +1.0)
- **Missing / NoData**: The value `-3000` is correctly encoded. 
- **Quality Variable**: `_250m_16_days_VI_Quality`. This bitmask contains vital QA flags regarding cloud cover, aerosol quantity, shadow, and snow/ice.
- **Physical Range**: Perfectly reasonable. The maximum scaled value is 1.0 (dense vegetation) and the minimum is -0.2 (water/barren).

---

## 6. LST Quality Audit
- **Main Variable**: `LST_Day_1km`
- **Scale Factor**: 0.02
- **Add Offset**: 0.0
- **_FillValue**: 0
- **Valid Range (Raw)**: 7500 to 65535
- **The 65535 Anomaly**: The metadata claims 65535 is a valid maximum. However, 65535 * 0.02 = 1310.7 Kelvin (~1037°C), which is physically impossible for the Earth's surface. **65535 is a legacy uint16 max value serving as an undocumented NoData/Error flag for missing data (e.g., dense cloud cover where LST cannot be retrieved)**. It MUST be masked alongside `0`.
- **Quality Variable**: `QC_Day`. This bitmask contains Mandatory QA flags (LST produced, LST not produced due to clouds), data quality flags, and emissivity error flags.
- **Sample Physical Stats**: Valid physical values for LST over Morocco (excluding 0 and 65535) range realistically from ~270K (-3°C in the Atlas Mountains) to ~330K (57°C in the Sahara during summer).

---

## 7. Temporal Consistency Check
- **NDVI**: 100% regular (16-day cadence).
- **LST**: 99.8% regular (8-day cadence). One observation period is missing. This gap must be interpolated during preprocessing.

---

## 8. Spatial Consistency Check
- The boundaries correctly encompass the requested territory (Morocco).
- **Masking**: The arrays represent a large rectangular bounding box. Since Morocco is irregularly shaped, a significant portion of the array covers the Atlantic Ocean and neighboring countries. A geospatial shapefile/mask of Morocco is required to clip the data before analysis, but no such mask exists in the raw files.

---

## 9 & 10. Spatial & Temporal Alignment Strategies
To combine LST (8-day, 1km) and NDVI (16-day, 250m):

### Temporal Strategy
- **Recommendation**: Downsample LST to 16-day. 
- **Why**: Averaging two 8-day LST composites perfectly aligns with the 16-day NDVI period and reduces noise/cloud contamination. Upsampling NDVI to 8-day would require guessing/interpolating data that doesn't exist.

### Spatial Strategy: A (NDVI -> 1km) vs B (LST -> 250m)
- **Recommendation**: **A. Aggregating NDVI to 1km (Downsampling)**
- **Scientific Justification**: Upsampling LST to 250m (B) creates false spatial precision. You cannot invent temperature variations at a 250m scale when the sensor only captured 1km blocks. Aggregating NDVI to 1km (A) reduces noise, preserves physical truth (by averaging vegetation within the 1km temperature cell), and massively reduces the dataset size for faster computation. 

---

## 11. Computational Requirements
- **NDVI Raw Array**: 271 (t) x 7224 (y) x 7206 (x) = ~14.1 Billion pixels.
- **NDVI Memory**: ~28.2 GB as `int16`. Over **100 GB** if naively converted to `float64` for processing.
- **LST Raw Array**: 530 (t) x 1807 (y) x 1802 (x) = ~1.7 Billion pixels.
- **LST Memory**: ~6.8 GB as `int32`.
- **Strategy**: We CANNOT process these in memory using standard pandas/numpy. We MUST use `xarray` backed by `dask` to chunk the arrays spatially (e.g., `chunks={'xdim': 1000, 'ydim': 1000, 'time': -1}`) and compute lazily. No extra packages are needed since `xarray` and `dask` usually handle this.

---

## PREPROCESSING READINESS

**READY WITH CONDITIONS**

**Why:** The NASA data files are successfully downloaded, mathematically sound, and formatted correctly. However, due to their massive size (~35 GB of combined RAM required for raw arrays), the presence of undocumented error values (`65535` in LST), missing temporal steps in LST, and differing grid resolutions, they require a highly optimized, chunked preprocessing pipeline.

---

## REQUIRED NEXT STEPS

Do NOT execute these yet. This is the blueprint for the preprocessing phase:

1. **Lazy Loading**: Open both files using `xarray.open_dataset(..., chunks=...)` to prevent MemoryErrors.
2. **Value Masking**: 
   - Mask NDVI where value == `-3000`.
   - Mask LST where value == `0` OR value == `65535`.
3. **Scaling**: Multiply by `scale_factor` to convert to physical units (NDVI float, Kelvin float).
4. **Temporal Alignment**: Resample LST by computing the temporal mean over 16-day intervals to match NDVI timestamps, ignoring NaNs.
5. **Spatial Alignment**: Coarsen/Aggregate the NDVI 250m grid to the LST 1km grid using spatial averaging.
6. **Masking & Cropping**: Use a GeoJSON/Shapefile of Morocco to apply a spatial mask, setting all pixels outside Morocco to NaN, and crop the bounding box to reduce array size.
7. **Storage**: Export the final combined, 1km, 16-day, masked array as an analysis-ready NetCDF file (`terra_pulse_morocco_clean.nc`).
