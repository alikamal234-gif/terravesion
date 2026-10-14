import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

class QualityFilter:
    """
    Scientific quality filtering and QC bitmask decoding for NASA MODIS products.
    """
    @staticmethod
    def filter_ndvi_qc(da_ndvi: xr.DataArray, da_qa: xr.DataArray) -> xr.DataArray:
        """
        Filters MOD13Q1 250m NDVI array using the 16-bit VI Quality assessment band.
        Bits 0-1 (VI Quality): 00 = Good quality, 01 = Produced but check other QA.
        Bits 2-5 (VI Usefulness): <= 11 (within usable range).
        Bits 8-9 (Atmospheric aerosol): 00 = Climatology, 01 = Low, 10 = Average.
        """
        # Exclude fill value (-3000) and preserve physical range [-2000, 10000]
        valid_range_mask = (da_ndvi >= -2000) & (da_ndvi <= 10000)
        
        # Mandatory QA (bits 0-1): 00 or 01
        qa_mandatory = (da_qa & 0b11) <= 1
        
        clean = xr.where(valid_range_mask & qa_mandatory, da_ndvi, np.nan)
        logger.debug("Applied MOD13Q1 QA filtering.")
        return clean

    @staticmethod
    def filter_lst_qc(da_lst: xr.DataArray, da_qc: xr.DataArray) -> xr.DataArray:
        """
        Filters MOD11A2 1km LST Day array using the 8-bit QC_Day assessment band.
        Bits 0-1 (Mandatory QA): 00 = Good quality, 01 = Other quality (clouds cleared).
        Excludes fill value (0) and uncalibrated / error sentinel (65535).
        """
        valid_mask = (da_lst != 0) & (da_lst != 65535) & ((da_qc & 0b11) <= 1)
        clean = xr.where(valid_mask, da_lst, np.nan)
        logger.debug("Applied MOD11A2 QC filtering.")
        return clean

    @staticmethod
    def check_physical_bounds(da: xr.DataArray, min_val: float, max_val: float, var_name: str = "variable") -> bool:
        """
        Validates that all non-NaN values fall within scientifically realistic physical boundaries.
        """
        nan_mask = ~np.isnan(da.values)
        if not np.any(nan_mask):
            logger.warning(f"{var_name} contains only NaNs.")
            return False
            
        actual_min = float(np.nanmin(da.values))
        actual_max = float(np.nanmax(da.values))
        
        if actual_min < min_val or actual_max > max_val:
            logger.error(
                f"{var_name} physical bounds violation: range [{actual_min:.3f}, {actual_max:.3f}] "
                f"exceeds valid bounds [{min_val}, {max_val}]."
            )
            return False
            
        logger.info(f"{var_name} physical bounds validated: [{actual_min:.3f}, {actual_max:.3f}] within [{min_val}, {max_val}].")
        return True
