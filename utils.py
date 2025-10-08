# Utility functions by Vit Ruzicka
# Might need: "pip install georeader"
# Also might need (for EMIT scenes download): running "auth = earthaccess.login()"
# TODO: feel free to change for your data loading scripts...

import earthaccess
# auth = earthaccess.login()
import shapely
import geopandas as gpd
import pylab as plt
import rasterio as rio
from georeader.readers import emit
import os
import os.path
from typing import Tuple, Optional, Any, Union, Dict
import rasterio
import rasterio.windows
import numpy as np
from georeader.geotensor import GeoTensor
import rasterio.warp
from datetime import datetime, timezone
import netCDF4

def file_exists(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def rio_load(path, verbose=False):
    with rio.open(path) as src:
        data = src.read()
        if verbose: print("crs", src.crs)
    return data

def vec_load(vector_path):
    gdf = gpd.read_file(vector_path)
    return gdf

def shapely_multipolygon_to_polygons(multipol):
    # If multipolygon, split
    if (isinstance(multipol, shapely.geometry.multipolygon.MultiPolygon)):
        return multipol.geoms
    # If single polygon give just that one
    if (isinstance(multipol, shapely.geometry.polygon.Polygon)):
        return [multipol]

def geopandas_to_shapely(gdf):
    # Convering between representations
    # from geopandas.geodataframe.GeoDataFrame (assuming just one entry)
    # to shapely.geometry.multipolygon.MultiPolygon
    # print(gdf.crs) # EPSG:32649 - I think this matches what's set up in the geotiff...
    vec_crs = "EPSG:4326" # this is the format we use everywhere
    gdf = gdf.to_crs(vec_crs)
    geometry_series = gdf.geometry
    shapely_polygon = geometry_series.iloc[0]
    return shapely_polygon

def how_many_pixels_does_polygon_occupy(polygon, reference_file="wmf.tif"):
    # potentially external dependencies ...
    from georeader.rasterio_reader import RasterioReader
    from georeader import rasterize

    ref_data = RasterioReader(reference_file)
    rasterized = rasterize.rasterize_geometry_like(polygon, data_like=ref_data, value=1, fill=0, crs_geometry="EPSG:4326")
    number_of_pixels = np.sum(rasterized.values)
    return number_of_pixels, rasterized.values


def get_rad_name(tile_name, with_file_type=False):
    if with_file_type: return tile_name+".nc"
    return tile_name
def get_obs_name(tile_name, with_file_type=False):
    name = tile_name.replace("_RAD_", "_OBS_")
    if with_file_type: return name+".nc"
    return name
def get_cmf_name(tile_name, with_file_type=False):
    name = tile_name.replace("_L1B_RAD_001_","_L2B_CH4ENH_002_")
    if with_file_type: return name+".tif"
    return name
def get_mask_name(tile_name, with_file_type=False):
    name = tile_name.replace("_L1B_RAD_", "_L2A_MASK_")
    if with_file_type: return name+".nc"
    return name

def download_granule(tile_name = "EMIT_L1B_RAD_001_20250529T024614_2514902_001", local_path = "granule_downloads",
                     also_download_l2a_mask = False,
                     also_download_official_cmf = False):
    auth = earthaccess.login(persist=True)
    print("authenticated with earthaccess:", auth.authenticated)

    # input is a list of granule links (HTTP)
    rad_name = tile_name
    obs_name = tile_name.replace("_RAD_","_OBS_")

    links = ["https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/"+rad_name+"/"+rad_name+".nc",
      "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/"+rad_name+"/"+obs_name+".nc"]

    if also_download_l2a_mask:
        rfl_name = tile_name.replace("_L1B_RAD_","_L2A_RFL_")
        mask_name = tile_name.replace("_L1B_RAD_","_L2A_MASK_")
        l2a_mask_link = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/"+rfl_name+"/"+mask_name+".nc"
        links.append(l2a_mask_link)

    if also_download_official_cmf:
        # EMITL2BCH4ENH.002/EMIT_L2B_CH4ENH_002_20230614T102451_2316507_028/EMIT_L2B_CH4ENH_002_20230614T102451_2316507_028.tif
        cmf_name = tile_name.replace("_L1B_RAD_001_","_L2B_CH4ENH_002_")
        cmf_link = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4ENH.002/"+cmf_name+"/"+cmf_name+".tif"
        links.append(cmf_link)

    # check if these files already exist?
    file_needed = [os.path.join(local_path, rad_name+".nc"), os.path.join(local_path, obs_name+".nc")]
    if also_download_l2a_mask: file_needed.append(os.path.join(local_path, mask_name + ".nc"))
    if also_download_official_cmf: file_needed.append(os.path.join(local_path, cmf_name + ".tif"))

    links_missing = []
    all_exist = True
    for idx, file in enumerate(file_needed):
        if not file_exists(file):
            links_missing.append(links[idx])
            all_exist = False
    if all_exist:
            print("Already dowloaded previously, skipping!")
            return True

    earthaccess.download(links_missing, local_path=local_path)
    print("Downloaded!")

### GENERAL NC LOADER:
# Same as the default one, except it supports general layer name (e.g.: "mask" for L2A data)

class NCImage(emit.EMITImage):
    def __init__(self, filename: str, glt: Optional[GeoTensor] = None,
                 band_selection: Optional[Union[int, Tuple[int, ...], slice]] = slice(None),
                 layer_name = 'radiance', skip_wavelenghts=False):
        self.layer_name = layer_name
        ###
        self.filename = filename
        self.nc_ds = netCDF4.Dataset(self.filename, 'r', format='NETCDF4')
        self._nc_ds_obs: Optional[netCDF4.Dataset] = None
        self._nc_ds_l2amask: Optional[netCDF4.Dataset] = None
        self._observation_bands = None
        self._mask_bands = None
        self.nc_ds.set_auto_mask(False)  # disable automatic masking when reading data
        # self.real_shape = (self.nc_ds['radiance'].shape[-1],) + self.nc_ds['radiance'].shape[:-1]

        self._mean_sza = None
        self._mean_vza = None
        self.obs_file: Optional[str] = None
        self.l2amaskfile: Optional[str] = None

        self.real_transform = rasterio.Affine(self.nc_ds.geotransform[1], self.nc_ds.geotransform[2], self.nc_ds.geotransform[0],
                                              self.nc_ds.geotransform[4], self.nc_ds.geotransform[5], self.nc_ds.geotransform[3])

        self.time_coverage_start = datetime.strptime(self.nc_ds.time_coverage_start, "%Y-%m-%dT%H:%M:%S%z")
        self.time_coverage_end = datetime.strptime(self.nc_ds.time_coverage_end, "%Y-%m-%dT%H:%M:%S%z")

        self.dtype = self.nc_ds[self.layer_name].dtype
        self.dims = ("band", "y", "x")
        self.fill_value_default = self.nc_ds[self.layer_name]._FillValue
        self.nodata = self.nc_ds[self.layer_name]._FillValue
        self.units = self.nc_ds[self.layer_name].units

        if glt is None:
            glt_arr = np.zeros((2,) + self.nc_ds.groups['location']['glt_x'].shape, dtype=np.int32)
            glt_arr[0] = np.array(self.nc_ds.groups['location']['glt_x'])
            glt_arr[1] = np.array(self.nc_ds.groups['location']['glt_y'])
            # glt_arr -= 1 # account for 1-based indexing

            # https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html
            self.glt = GeoTensor(glt_arr, transform=self.real_transform,
                                 crs=rasterio.crs.CRS.from_wkt(self.nc_ds.spatial_ref),
                                 fill_value_default=0)
        else:
            self.glt = glt

        self.valid_glt = np.all(self.glt.values != self.glt.fill_value_default, axis=0)
        xmin, ymin, xmax, ymax = self._bounds_indexes_raw()  # values are 1-based!

        # glt has the absolute indexes of the netCDF object
        # glt_relative has the relative indexes
        self.glt_relative = self.glt.copy()
        self.glt_relative.values[0, self.valid_glt] -= xmin
        self.glt_relative.values[1, self.valid_glt] -= ymin

        self.window_raw = rasterio.windows.Window(col_off=xmin - 1, row_off=ymin - 1,
                                                  width=xmax - xmin + 1, height=ymax - ymin + 1)

        self.band_selection = band_selection

        if not skip_wavelenghts:
            if "wavelengths" in self.nc_ds['sensor_band_parameters'].variables:
                self.bandname_dimension = "wavelengths"
            elif "radiance_wl" in self.nc_ds['sensor_band_parameters'].variables:
                self.bandname_dimension = "radiance_wl"
            else:
                raise ValueError(f"Cannot find wavelength dimension in {list(self.nc_ds['sensor_band_parameters'].variables.keys())}")

            self.wavelengths = self.nc_ds['sensor_band_parameters'][self.bandname_dimension][self.band_selection]
            self.fwhm = self.nc_ds['sensor_band_parameters']['fwhm'][self.band_selection]
            self._observation_date_correction_factor: Optional[float] = None

    def load_raw(self, transpose: bool = True) -> np.array:
        slice_y, slice_x = self.window_raw.toslices()
        if isinstance(self.band_selection, slice):
            data = np.array(self.nc_ds[self.layer_name][slice_y, slice_x, self.band_selection])
        else:
            data = np.array(self.nc_ds[self.layer_name][slice_y, slice_x][..., self.band_selection])
        # transpose to (C, H, W)
        if transpose and (len(data.shape) == 3):
            data = np.transpose(data, axes=(2, 0, 1))
        return data

    def load(self, boundless:bool=True, as_reflectance:bool=False, orthorectify:bool=True)-> GeoTensor:
        data = self.load_raw() # (C, H, W) or (H, W)
        if as_reflectance:
            invalids = np.isnan(data) | (data == self.fill_value_default)
            from georeader import reflectance

            thuiller = reflectance.load_thuillier_irradiance()
            response = reflectance.srf(self.wavelengths, self.fwhm, thuiller["Nanometer"].values)
            solar_irradiance_norm = thuiller["Radiance(mW/m2/nm)"].values.dot(response) / 1_000
            data = reflectance.radiance_to_reflectance(data, solar_irradiance_norm,
                                                       units=self.units,
                                                       observation_date_corr_factor=self.observation_date_correction_factor)
            data[invalids] = self.fill_value_default

        if orthorectify:
            return self.georreference(data, fill_value_default=self.fill_value_default)
        else:
            return GeoTensor(values=data, transform=self.transform, crs=self.crs,
                     fill_value_default=self.fill_value_default)
