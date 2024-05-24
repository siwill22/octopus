import pygmt
import pygplates
import numpy as np
import geopandas as gpd
import os, sys
import xarray as xr
import pyshtools
from xrspatial import proximity

from ptt.utils.call_system_command import call_system_command
import gprm.utils.paleogeography as pg
from gprm.utils import inpaint
from gprm.utils.deformation import topological_reconstruction

sys.path.append('/Users/simon/GIT/degenerative_art/')
import map_effects as me

sys.path.append('../sphipple/')
from stipple_scalar_grids_spherical import quantize_grid, mach_banding, write_netcdf_grid


def generate_seafloor_bathymetry(filename, final_grd_sampling):
    
    seafloor_age = pygmt.grdsample(filename,
                                   region='d', spacing='{:f}d'.format(final_grd_sampling))

    seafloor_depth = seafloor_age.copy(deep=True)
    seafloor_depth.data = pg.age2depth(seafloor_depth.data, model='GDH1')

    return seafloor_depth



def generate_land_topography(reconstruction_model, reconstruction_time,
                             final_grd_sampling,
                             min_distance_to_coastlines=0,
                             max_distance_to_trenches=1200000,
                             orogeny_geometries=None):
    
    rp = me.reconstruct_and_rasterize_polygons(reconstruction_model.continent_polygons[0], 
                                               reconstruction_model.rotation_model, 
                                               reconstruction_time=reconstruction_time, 
                                               sampling=final_grd_sampling)

    _, prox_ocean = me.raster_buffer(rp, inside='both')

    # Get subduction zones, and compute a raster of distances to the nearest one
    snapshot = reconstruction_model.plate_snapshot(reconstruction_time)

    szs = snapshot.get_boundary_features(boundary_types=['subduction'])
    all_sz_points = []
    for sz in szs:
        if sz.get_geometry():
            all_sz_points.extend(sz.get_geometry().to_tessellated(np.radians(0.1)).to_lat_lon_list())

    prox_sz = me.points_proximity(x=[lon for lat,lon in all_sz_points],
                                  y=[lat for lat,lon in all_sz_points],
                                  spacing=final_grd_sampling)
    
    # combine the rasters to isolate areas not too near to coastlines
    # and not too far from subduction zones
    m1 = prox_ocean.where((prox_ocean>min_distance_to_coastlines))
    m2 = prox_sz.where(prox_sz<max_distance_to_trenches)
    m2.data = m1.data+m2.data

    m2.data[np.isnan(m2.data)] = -999
    mountain_core = proximity(m2, target_values=[-999], distance_metric='GREAT_CIRCLE')
    
    land = prox_ocean.where(prox_ocean==0, 1)

    topography = (mountain_core/200)+(land*200)
    
    
    # Orogeny Points
    if orogeny_geometries is not None:
        orogeny_points = []
        pygplates.reconstruct(orogeny_geometries, 
                              reconstruction_model.rotation_model, orogeny_points, reconstruction_time)
        all_op_points = []
        for op in orogeny_points:
            if op.get_reconstructed_geometry():
                all_op_points.extend(op.get_reconstructed_geometry().to_tessellated(np.radians(0.1)).to_lat_lon_list())
        prox_orogeny = me.points_proximity(x=[lon for lat,lon in all_op_points],
                                           y=[lat for lat,lon in all_op_points],
                                           spacing=final_grd_sampling)

        m3 = prox_orogeny.where(prox_orogeny<max_distance_to_trenches)
        m3.data = m1.data+m3.data

        m3.data[np.isnan(m3.data)] = -999
        orogeny_core = proximity(m3, target_values=[-999], distance_metric='GREAT_CIRCLE')
        
        topography += (orogeny_core/150)
        
    else:
        orogeny_core = None


    topography = topography.where(topography>0, np.nan)
    
    
    return land, mountain_core, orogeny_core, topography


def merge_topography_and_bathymetry(seafloor_depth, topography, prox_sz=None,
                                    trench_dist_max = 300e3, trench_depth_max = 3000.):
    
    # recall that the coordinate names are inconsistent (x,y versus lat/lon)
    merge = seafloor_depth.copy(deep=True)
    merge.data = topography.data
    merge = merge.where(np.isfinite(merge), seafloor_depth)#.data[np.isnan(topography.data)] = seafloor_depth.data[np.isnan(topography.data)]

    #merge = pygmt.grdfill(merge, mode='s0.8', verbose='q')

    #bm = pygmt.grd2xyz(merge).dropna().reset_index(drop=True)
    #print('spherical interpolation step.....')
    #merge = pygmt.sphinterpolate(data=bm, spacing=final_grd_sampling, region='d', Q=0)

    #from rasterio.fill import fillnodata
    #merge.data = fillnodata(merge.data, mask=np.isnan(merge.data), smoothing_iterations=10) 
    merge.data = inpaint.fill_ndimage(merge.data)

    if prox_sz is not None:
        subduction_trench = 1-(prox_sz-trench_dist_max)
        subduction_trench = (subduction_trench.where(subduction_trench>0, 0)/trench_dist_max)*trench_depth_max
        subduction_trench = subduction_trench.where(np.isnan(topography), 0)

        merge.data = merge.data-subduction_trench
    
    return merge



def add_seamount_trails(reconstruction_model, 
                        elevation_map,
                        hot_spot_points, 
                        final_grd_sampling,
                        anchor_plate_id = 0, 
                        initial_time = 100, 
                        youngest_time = 0, 
                        time_increment = 1):
    
    topological_model = pygplates.TopologicalModel(reconstruction_model.dynamic_polygons,
                                                   reconstruction_model.rotation_model,
                                                   anchor_plate_id=anchor_plate_id)
    
    hot_spot_trail = {}
  
    #reconstruction_times = np.arange(100,-1,-10)
    for hot_spot_key in hot_spot_points:
        rp = []
        ra = []
        reconstruction_times = np.sort(np.random.uniform(0, 200, size=25))
        for reconstruction_time in reconstruction_times:

            reconstructed_points = topological_model.reconstruct_geometry(
                pygplates.PointOnSphere(hot_spot_points[hot_spot_key]),
                initial_time=reconstruction_time,
                oldest_time=200.,
                youngest_time=0.,
                time_increment=1)

            #print(reconstruction_time, reconstructed_points.get_geometry_points(0.))
            if reconstructed_points.get_geometry_points(0.):
                rp.append(reconstructed_points.get_geometry_points(0.)[0].to_lat_lon())
                ra.append(reconstruction_time)

        lonlats = list(zip(*rp))
        hot_spot_trail[hot_spot_key] = gpd.GeoDataFrame(data={'age': ra},
                                              geometry=gpd.points_from_xy(lonlats[1], lonlats[0]), 
                                              crs=4326)
        
    for hot_spot_key in hot_spot_trail.keys():

        d2sm = me.points_proximity(hot_spot_trail[hot_spot_key].geometry.x, 
                                   hot_spot_trail[hot_spot_key].geometry.y, spacing=final_grd_sampling)
        d2sm = d2sm.where(d2sm>100.,100.)
        d2sm = (1./d2sm)


        filter_length_km = 1000
        #pygmt.grdfilter(tmp, filter='g{:f}k+h'.format(filter_length_km), 
        #                distance='2', coltypes='g').plot()
        d2sm.to_netcdf('./_tmp.nc')
        call_system_command(['gmt', 'grdfilter', '_tmp.nc', 
                             '-Fg{:f}k'.format(filter_length_km), '-D2', '-G_tmpp.nc', '-fg'])
        seamounts = xr.open_dataarray('./_tmpp.nc')
        os.remove('./_tmpp.nc')
        #"""

        seamounts = (seamounts/seamounts.data.max())*np.random.uniform(low=2000,high=3000,size=1)
        seamounts = seamounts.where(elevation_map<0, 0)
        elevation_map+=seamounts
        
    return elevation_map, hot_spot_trail



def make_noise_grid(lmax=300, exponent=-2, scaling=1, spacing='6m'):
    degrees = np.arange(lmax+1, dtype=float)
    degrees[0] = np.inf

    power_per_degree = degrees**(exponent)
    power_per_degree[:200] = 0

    noise = pyshtools.SHCoeffs.from_random(power_per_degree, seed=None).expand().to_xarray()

    noise = pygmt.grdsample(noise, region='d', spacing=spacing)
    return noise*scaling


def generate_seafloor_fabric(seafloor_age, final_grd_sampling, 
                             q=5, noise_scaling=5,
                             hp_filter_length_km = 500., 
                             lp_filter_length_km = 250.):
    

    #agegrid = xr.open_dataarray('/Users/simon/Data/AgeGrids/2020/age.2020.1.GeeK2007.6m.nc')
    #agegrid = pygmt.grdsample('../octopus/Atlantis2/masked/Atlantis2_seafloor_age_mask_0.0Ma.nc',
    #                          region='d', spacing='{:f}d'.format(final_grd_sampling))
    #agegrid.data += make_noise_grid(lmax=400, scaling=2.).data


    quantized_raster = quantize_grid(seafloor_age, q=q)

    quantized_raster.data += make_noise_grid(lmax=400, scaling=noise_scaling, spacing='{:f}d'.format(final_grd_sampling)).data

    quantized_raster

    write_netcdf_grid('qraster.nc', seafloor_age.lon.data, seafloor_age.lat.data, quantized_raster)
    #filt_grid = pygmt.grdfilter(quantized_raster, 
    #                            filter='g{:f}k+h'.format(filter_length_km), 
    #                            distance='2', coltypes='g')

    call_system_command(['gmt', 'grdfilter', 'qraster.nc', 
                         '-Fg{:f}k+h'.format(hp_filter_length_km), '-D2', '-Gmach_banded_raster.nc', '-fg'])
    call_system_command(['gmt', 'grdfilter', 'mach_banded_raster.nc', 
                         '-Fg{:f}k'.format(lp_filter_length_km), '-D2', '-Gmach_banded_raster.nc', '-fg'])
    filt_grid = xr.open_dataarray('./mach_banded_raster.nc')
    os.remove('mach_banded_raster.nc')

    filt_grid = filt_grid.where(np.isfinite(filt_grid), 0)
    
    return filt_grid
    
    
    