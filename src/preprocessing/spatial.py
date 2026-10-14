import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterio.enums import Resampling
import rioxarray
import logging

logger = logging.getLogger(__name__)

class SpatialAligner:
    """
    Handles spatial alignment and reprojection of datasets.
    """
    @staticmethod
    def create_geodataframe(df: pd.DataFrame, lat_col: str, lon_col: str, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Converts a Pandas DataFrame to a GeoDataFrame.
        """
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"Columns {lat_col} and {lon_col} must be present in the DataFrame.")
            
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs
        )
        logger.info(f"Created GeoDataFrame with CRS {crs}")
        return gdf

    @staticmethod
    def align_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
        """
        Reprojects a GeoDataFrame to a target CRS.
        """
        if gdf.crs == target_crs:
            return gdf
        
        reprojected = gdf.to_crs(target_crs)
        logger.info(f"Reprojected GeoDataFrame from {gdf.crs} to {target_crs}")
        return reprojected

    @staticmethod
    def align_raster_to_target(source_da: xr.DataArray, target_da: xr.DataArray, resampling_method=Resampling.average) -> xr.DataArray:
        """
        Geospatially resamples and aligns a source raster DataArray to a target raster DataArray.
        """
        aligned = source_da.rio.reproject_match(target_da, resampling=resampling_method)
        logger.info(f"Resampled raster from {source_da.shape} to match target {target_da.shape}")
        return aligned

    @staticmethod
    def mask_and_crop_raster(da: xr.DataArray, boundary_gdf: gpd.GeoDataFrame) -> xr.DataArray:
        """
        Clips and masks a raster DataArray with a GeoDataFrame geometry.
        Pixels outside the geometry become NaN, and the array is cropped to the bounding box.
        """
        # Ensure CRS matches
        raster_crs = da.rio.crs
        if boundary_gdf.crs != raster_crs:
            boundary_gdf = boundary_gdf.to_crs(raster_crs)
            
        clipped = da.rio.clip(boundary_gdf.geometry, drop=True)
        logger.info(f"Clipped raster to boundary. New shape: {clipped.shape}")
        return clipped
