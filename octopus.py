import pygmt
import pygplates
import numpy as np
import geopandas as gpd
import os, sys
import xarray as xr
import pyshtools
from xrspatial import proximity
from scipy.interpolate import interpn

from scipy.ndimage import convolve


from ptt.utils.call_system_command import call_system_command
import gprm.utils.paleogeography as pg
from gprm.utils import inpaint
from gprm.utils.deformation import topological_reconstruction
from gprm import PointDistributionOnSphere

sys.path.append('/Users/simon/GIT/degenerative_art/')
import map_effects as me

sys.path.append('../sphipple/')
from stipple_scalar_grids_spherical import quantize_grid, mach_banding, write_netcdf_grid

sys.path.append('/Users/simon/GIT/pgpslabs/')
import slab_tracker_utils as slab

sys.path.append('/Users/simon/GIT/vh0/notebooks/ModelGeneration/')
import ocean_remanence_vectors as orv
from remit.data.models import create_vim
from remit.utils.grid import coeffs2map
#from remit.utils.profile import polarity_timescale
from remit.earthvim import SeafloorGrid, GlobalVIS, PolarityTimescale
from remit.utils.grid import shmaggrid2tmi




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
    merge.data = inpaint.fill_inpaint(merge.data, kernel_size=7)# method='idw')

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
                        filter_length_km = 500,
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



def make_noise_grid(lmin=0, lmax=300, exponent=-2, scaling=1, spacing='6m'):
    
    degrees = np.arange(lmax+1, dtype=float)
    degrees[0] = np.inf

    power_per_degree = degrees**(exponent)
    power_per_degree[:lmin] = 0

    noise = pyshtools.SHCoeffs.from_random(power_per_degree, seed=None).expand().to_xarray()

    noise = pygmt.grdsample(noise, region='d', spacing=spacing)
    return noise*scaling


def scale_array_to_range(array, new_min=-1000, new_max=1000):
    """
    Scales a 2D numpy array to a specified range [-1000, 1000].

    Parameters:
        array (numpy.ndarray): Input 2D array.
        new_min (float): The minimum value of the new range (default -1000).
        new_max (float): The maximum value of the new range (default +1000).

    Returns:
        numpy.ndarray: Scaled array with values in the specified range.
    """
    old_min = np.min(array)
    old_max = np.max(array)
    
    # Avoid division by zero if all values in the array are the same
    if old_min == old_max:
        return np.full_like(array, (new_max + new_min) / 2)
    
    # Calculate the scaling factor and offset
    scale_factor = (new_max - new_min) / (old_max - old_min)
    offset = new_min - (old_min * scale_factor)
    
    # Apply the scaling factor and offset
    scaled_array = (array * scale_factor) + offset
        
    return scaled_array


def generate_seafloor_fabric(seafloor_age, final_grd_sampling, 
                             target_range=None,
                             q=5, noise_scaling=5,
                             hp_filter_length_km = 500., 
                             lp_filter_length_km = 250.):

    quantized_raster = quantize_grid(seafloor_age, q=q)

    quantized_raster.data += make_noise_grid(lmin=200, lmax=400, scaling=noise_scaling, spacing='{:f}d'.format(final_grd_sampling)).data

    write_netcdf_grid('qraster.nc', seafloor_age.lon.data, seafloor_age.lat.data, quantized_raster)

    call_system_command(['gmt', 'grdfilter', 'qraster.nc', 
                         '-Fg{:f}k+h'.format(hp_filter_length_km), '-D2', '-Gmach_banded_raster.nc', '-fg'])
    call_system_command(['gmt', 'grdfilter', 'mach_banded_raster.nc', 
                         '-Fg{:f}k'.format(lp_filter_length_km), '-D2', '-Gmach_banded_raster.nc', '-fg'])
    filt_grid = xr.open_dataarray('./mach_banded_raster.nc')
    os.remove('mach_banded_raster.nc')

    filt_grid = filt_grid.where(np.isfinite(filt_grid), 0)
    
    if target_range is not None:
        filt_grid.data = scale_array_to_range(filt_grid.data, new_min=-target_range/2, new_max=target_range/2)
        
    filt_grid = filt_grid - filt_grid.data.mean()
        
    return filt_grid


# Functions for spatially-varying smoothing
def create_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    k = (size - 1) // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()


def apply_spatially_varying_smoothing(smoothing_array, data_array, vary_kernel_size=False):
    """Apply spatially varying smoothing to the data_array based on smoothing_array."""
    smoothed_array = np.zeros_like(data_array)
    rows, cols = data_array.shape
    
    for i in range(rows):
        for j in range(cols):
            # Determine the kernel size and sigma based on the smoothing factor
            smoothing_factor = smoothing_array[i, j]
            if vary_kernel_size:
                kernel_size = max(3, int(smoothing_factor) * 2 + 1)  # Ensure kernel size is at least 3
            else:
                kernel_size = 7
            sigma = smoothing_factor
            
            # Create the Gaussian kernel
            kernel = create_gaussian_kernel(kernel_size, sigma)
            
            # Determine the region to apply the kernel
            half_k = kernel_size // 2
            i_min = max(i - half_k, 0)
            i_max = min(i + half_k + 1, rows)
            j_min = max(j - half_k, 0)
            j_max = min(j + half_k + 1, cols)
            
            # Extract the region and apply the convolution
            region = data_array[i_min:i_max, j_min:j_max]
            k_i_min = max(half_k - i, 0)
            k_i_max = min(half_k + (rows - i), kernel_size)
            k_j_min = max(half_k - j, 0)
            k_j_max = min(half_k + (cols - j), kernel_size)
            region_kernel = kernel[k_i_min:k_i_max, k_j_min:k_j_max]
            
            # Apply convolution to the region and assign the result to the center cell
            if region.size > 0:
                smoothed_value = np.sum(region * region_kernel)
                smoothed_array[i, j] = smoothed_value
    
    return smoothed_array    
    
    
def smooth_seafloor(seafloor_age, seafloor_depth, scaling=0.1, target_range=None,
                    smoothing_age_min=30, smoothing_age_max=200, vary_kernel_size=False):    
    
    smoothed_array = seafloor_depth.copy(deep=True)
    
    smoothing_array = inpaint.fill_ndimage(seafloor_age.data)
    data_array = inpaint.fill_ndimage(seafloor_depth.data)

    smoothing_array[smoothing_array<smoothing_age_min] = smoothing_age_min
    smoothing_array[smoothing_array>smoothing_age_max] = smoothing_age_max

    smoothing_array = smoothing_array - smoothing_age_min + 0.0001
    smoothing_array = smoothing_array * scaling
    
    if vary_kernel_size:
        print('max kernel size = {:d}'.format(max(3, int(smoothing_array.max()) * 2 + 1)))
    smoothed_array.data = apply_spatially_varying_smoothing(smoothing_array, data_array, vary_kernel_size=vary_kernel_size)
    
    if target_range is not None:
        smoothed_array.data = scale_array_to_range(smoothed_array.data, new_min=-target_range/2, new_max=target_range/2)
    smoothed_array = smoothed_array - smoothed_array.data.mean()

    return smoothed_array


    
    
def generate_slab_earthquakes(reconstruction_model, 
                              n_samples=200,
                              start_time = 40.,
                              end_time = 0.,
                              time_step = 2.0,
                              dip_angle_degrees = 45.0,
                              line_tessellation_distance = np.radians(1.0)):
    
    agegrid_filename = None
    topology_features = reconstruction_model.dynamic_polygons
    rotation_model = reconstruction_model.rotation_model

    subduction_boundary_sections = slab.getSubductionBoundarySections(
        topology_features,
        rotation_model,
        0.)
    
    output_data = []
    dip_angle_radians = np.radians(dip_angle_degrees)
    time_list = np.arange(start_time,end_time-time_step,-time_step)

    for time in time_list:

        print('time %0.2f Ma' % time)

        # call function to get subduction boundary segments
        subduction_boundary_sections = slab.getSubductionBoundarySections(topology_features,
                                                                          rotation_model,
                                                                          time)

        # Set up an age grid interpolator for this time, to be used
        # for each tessellated line segment
        ##grdfile = '../agegrid-0.1/grid_files/unmasked/M16_seafloor_age_0.0Ma.nc'
        ##lut = slab.make_age_interpolator(grdfile)

        # Loop over each segment
        for segment_index,subduction_segment in enumerate(subduction_boundary_sections):

            # find the overrding plate id (and only continue if we find it)
            overriding_and_subducting_plates = slab.find_overriding_and_subducting_plates(subduction_segment,time)

            if not overriding_and_subducting_plates:
                continue
            overriding_plate, subducting_plate, subduction_polarity = overriding_and_subducting_plates

            overriding_plate_id = overriding_plate.get_resolved_feature().get_reconstruction_plate_id()
            subducting_plate_id = subducting_plate.get_resolved_feature().get_reconstruction_plate_id()

            subducting_plate_disappearance_time = -1.

            tessellated_line = subduction_segment.get_resolved_geometry().to_tessellated(line_tessellation_distance)

            #print len(tessellated_line.get_points())

            if agegrid_filename is not None:
                x = tessellated_line.to_lat_lon_array()[:,1]
                y = tessellated_line.to_lat_lon_array()[:,0]
                subduction_ages = lut.ev(np.radians(y+90.),np.radians(x+180.))
            else:
                # if no age grids, just fill the ages with zero
                subduction_ages = [0. for point in tessellated_line.to_lat_lon_array()[:,1]]

            # CALL THE MAIN WARPING FUNCTION
            (points, 
             point_depths, 
             polyline) = slab.warp_subduction_segment(tessellated_line,
                                                      rotation_model,
                                                      subducting_plate_id,
                                                      overriding_plate_id,
                                                      subduction_polarity,
                                                      time,
                                                      end_time,
                                                      time_step,
                                                      dip_angle_radians,
                                                      subducting_plate_disappearance_time)

            output_data.append(gpd.GeoDataFrame(
                data={'subduction_time':np.ones(len(point_depths)).T*time,
                      'depth': point_depths,
                      'age_at_subduction': subduction_ages},
                geometry=gpd.points_from_xy(polyline.to_lat_lon_array()[:,1], 
                                            polyline.to_lat_lon_array()[:,0]), 
                crs=4326))

    present_day_slab = gpd.GeoDataFrame(
        gpd.pd.concat(output_data, ignore_index=True), crs=output_data[0].crs)
    slab_earthquakes = present_day_slab.sample(n_samples)
    
    return gpd.GeoDataFrame(slab_earthquakes)


def generate_plate_boundary_earthquakes(reconstruction_model, n_points, 
                                    max_distance=2e4, min_depth=1, max_depth=60):
    
    snapshot = reconstruction_model.plate_snapshot(0.)
    plate_boundaries = snapshot.get_boundary_features()

    all_pb_points = []
    for pb in plate_boundaries:
        if pb.get_geometry():
            all_pb_points.extend(pb.get_geometry().to_tessellated(np.radians(0.1)).to_lat_lon_list())

    prox_pb = me.points_proximity(x=[lon for lat,lon in all_pb_points],
                                  y=[lat for lat,lon in all_pb_points],
                                  spacing=0.1)
    
    pts = PointDistributionOnSphere(N=int(n_points))

    plate_boundary_earthquakes = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts.longitude, pts.latitude), crs=4326)
    plate_boundary_earthquakes['distance_to_boundary'] = interpolate_values(prox_pb, plate_boundary_earthquakes)

    plate_boundary_earthquakes = plate_boundary_earthquakes.query('distance_to_boundary<@max_distance').reset_index()

    plate_boundary_earthquakes['depth'] = np.random.uniform(min_depth, max_depth, len(plate_boundary_earthquakes))
    
    return plate_boundary_earthquakes


def interpolate_values(dataarray, coordinates):
    """
    Interpolate values from a 2D xarray DataArray at arbitrary coordinates using linear interpolation.

    Parameters:
    dataarray (xarray.DataArray): 2D DataArray with coordinates as longitude and latitude.
    points (list of tuples): List of (longitude, latitude) coordinates where interpolation is desired.

    Returns:
    np.ndarray: Interpolated values at the specified points.
    """
    # Extract the coordinates and data values from the DataArray
    lon = dataarray['x'].values
    lat = dataarray['y'].values
    values = dataarray.values

    # Create the grid of points from the coordinates
    grid = (lat, lon)

    points = [(row.geometry.y, row.geometry.x) for i,row in coordinates.iterrows()]
    
    # Perform the interpolation
    interpolated_values = interpn(grid, values, points, method='nearest', bounds_error=False, fill_value=None)

    return interpolated_values



####################################################################
def generate_magnetic_map(reconstruction_model, age_grid_filename,
                          final_grd_sampling,
                          lmin=16, lmax=500, grid_dims=(1801,3601)):
    
    snapshot = reconstruction_model.plate_snapshot(0.)

    static_polygon_filename = './_tmp.shp'
    fc = pygplates.FeatureCollection([rt.get_resolved_feature() for rt in snapshot.resolved_topologies])
    fc.write(static_polygon_filename)

    age_grid, plate_id_raster = orv.build_input_grids(age_grid_filename,
                                                      static_polygon_filename,
                                                      final_grd_sampling)

    print('Generating magnetization vectors...')
    (paleo_latitude,
     paleo_declination) = orv.reconstruct_agegrid_to_birthtime(
        reconstruction_model, 
        age_grid, 
        plate_id_raster, 
        return_type='xarray')


    GPTS = PolarityTimescale(timescalefile='/Users/simon/Documents/2022IMAS-OUC_SOMG/PracFiles/Atlantis2/Atlantis_GPTS.txt')

    GK07 = {'seafloor_layer':'2d',
            'layer_boundary_depths':[0,500,1500,6500], 
            'layer_weights':[5,2.3,1.2], 
            'MagMax':None, 
            'P':5, 
            'lmbda':3, 
            'Mtrm':1, 
            'Mcrm':0,
            'PolarityTimescale':GPTS}


    ocean = SeafloorGrid.from_xarray(age_grid, paleo_declination, paleo_latitude)
    ocean.resample(shape=grid_dims)

    layer_params = GK07.copy()

    vis = GlobalVIS.from_random(exponent=-1.5, scaling=0.02)
    vis.resample(shape=grid_dims)

    print('Computing VIM...')
    totalvim = create_vim(ocean, vis, **layer_params)
    
    print('Computing vsh transform...')
    vsh, coeffs = totalvim.transform(lmax=lmax)

    print('Computing TMI map...')
    # Need to make this into xarray
    tmi = shmaggrid2tmi(coeffs.expand(a=6371000+50000.))

    return tmi
    

    
####################################################################
from scipy.stats import truncnorm

def generate_random_coordinates_by_latitude(n_points, central_lat=20, lat_std=5, lon_range=(-200, 200)):
    """
    Generate random coordinates on the surface of a sphere with latitude following a Gaussian distribution.
    
    Parameters:
    n_points (int): Number of random points to generate.
    central_lat (float): Central latitude for Gaussian distribution (in degrees).
    lat_std (float): Standard deviation for the Gaussian distribution (in degrees).
    lon_range (tuple): Range of longitude values (in degrees), typically (0, 360).
    
    Returns:
    list of tuples: List containing the generated (latitude, longitude) pairs.
    """
    # Define the truncated normal distribution for latitude
    lat_min, lat_max = central_lat - (lat_std*3), central_lat + (lat_std*3)  # 15 degrees above and below the central latitude
    a, b = (lat_min - central_lat) / lat_std, (lat_max - central_lat) / lat_std
    lat_distribution = truncnorm(a, b, loc=central_lat, scale=lat_std)
    
    # Generate latitudes
    latitudes = lat_distribution.rvs(n_points)
    
    # Randomly assign half of the points to the southern hemisphere
    southern_hemisphere = np.random.choice([True, False], size=n_points)
    latitudes[southern_hemisphere] *= -1
    
    # Generate longitudes uniformly
    longitudes = np.random.uniform(lon_range[0], lon_range[1], n_points)
    
    # Combine latitudes and longitudes
    #coordinates = list(zip(longitudes, latitudes))
    
    df = gpd.pd.DataFrame(data={'Longitude': longitudes,
                                'Latitude': latitudes})
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs=4326)
    

def generate_climate_sensitive_deposits(land_raster, central_lat, lat_std, n_points=100):

    coordinates = generate_random_coordinates_by_latitude(n_points, central_lat=central_lat, lat_std=lat_std)

    # Interpolate values at the specified points
    coordinates['interpolated_values'] = interpolate_values(land_raster, coordinates)
    #coordinates = coordinates.query('interpolated_values>0')
    
    return coordinates


def generate_weighted_random_points(dataarray, n_points):
    """
    Generate random points weighted by the values in a 2D xarray DataArray.

    Parameters:
    dataarray (xarray.DataArray): 2D DataArray with likelihood values at each grid node.
    n_points (int): Number of random points to generate.

    Returns:
    list of tuples: List containing the generated (x, y) coordinates.
    """
    # Normalize the DataArray values to probabilities
    values = dataarray.values
    probabilities = values / values.sum()

    # Flatten the probabilities and generate the CDF
    flattened_probabilities = probabilities.ravel()
    cdf = np.cumsum(flattened_probabilities)

    # Generate uniform random samples
    random_samples = np.random.rand(n_points)

    # Map the uniform random samples to the CDF
    random_indices = np.searchsorted(cdf, random_samples)

    # Convert the flat indices back to 2D indices
    y_indices, x_indices = np.unravel_index(random_indices, dataarray.shape)

    # Generate the corresponding x, y coordinates
    x_coords = dataarray['x'].values[x_indices]
    y_coords = dataarray['y'].values[y_indices]

    # Combine x and y coordinates into tuples
    #coordinates = list(zip(x_coords, y_coords))
    
    df = gpd.pd.DataFrame(data={'Longitude': x_coords,
                                'Latitude': y_coords})
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs=4326)



def unreconstruct_geodataframe(gdf, reconstruction_model, reconstructed_polygons, reconstruction_time):

    gdf = gdf.overlay(reconstructed_polygons, how='intersection', keep_geom_type=False)
    return reconstruction_model.reconstruct(gdf, reconstruction_time=reconstruction_time, reverse=True)

