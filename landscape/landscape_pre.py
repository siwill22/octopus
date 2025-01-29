import numpy as np
import xarray as xr
import uxarray as uxr

# On Docker turn off the warning on PROJ by specifying the PROJ lib path (uncomment the following line)
#os.environ['PROJ_LIB'] = '/opt/conda/envs/gospl/share/proj'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from scripts import umeshFcts as ufcts

widthCell = 30
input_path = "input_"+str(widthCell) 

# Build the mesh
ufcts.buildGlobalMeshSimple(widthCell, input_path)


# Loading the nc regular file
#ncgrid = xr.open_dataset('data/250.nc')
ncgrid = xr.open_dataset('data/topography_bathymetry.nc')
ncgrid = ncgrid.rename_vars({'z':'h'})
ncgrid['vx'] = (('lat','lon'),np.zeros(ncgrid.h.data.shape))
ncgrid['vy'] = (('lat','lon'),np.zeros(ncgrid.h.data.shape))
ncgrid['vz'] = (('lat','lon'),np.zeros(ncgrid.h.data.shape))
ncgrid['rain'] = (('lat','lon'),np.ones(ncgrid.h.data.shape))

#ncgrid

ncgrid = ncgrid[['h','vx','vy','vz','rain']]
#ncgrid

# Loading the UGRID file
ufile = input_path+'/mesh_'+str(widthCell)+'km.nc'
ugrid = uxr.open_grid(ufile) 
# ugrid

# Perform the interpolation (bilinear) 
var_path = 'vars_'+str(widthCell)
var_name = 'step_250'
ufcts.inter2UGRID(ncgrid,ugrid,var_path,var_name,type='face')



data_file = [var_path+'/'+var_name+'.nc']

# Get the information related to the mesh: primal and dual mesh
primal_mesh = uxr.open_dataset(ufile, *data_file, use_dual=False)
dual_mesh = uxr.open_dataset(ufile, *data_file, use_dual=True)

# Extract nodes and faces information
ucoords = np.empty((dual_mesh.uxgrid.n_node,3))
ucoords[:,0] = dual_mesh.uxgrid.node_x.values
ucoords[:,1] = dual_mesh.uxgrid.node_y.values
ucoords[:,2] = dual_mesh.uxgrid.node_z.values
ufaces = primal_mesh.uxgrid.node_face_connectivity.values

# Get information about your mesh:
print("Number of nodes: ",len(ucoords)," | number of faces ",len(ufaces))
edge_min = np.round(dual_mesh.uxgrid.edge_node_distances.min().values/1000.+0.,2)
edge_max = np.round(dual_mesh.uxgrid.edge_node_distances.max().values/1000.+0.,2)
edge_mean = np.round(dual_mesh.uxgrid.edge_node_distances.mean().values/1000.+0.,2)
print("edge range (km): min ",edge_min," | max ",edge_max," | mean ",edge_mean)




# Save voronoi mesh for visualisation purposes
saveVoro = False

if saveVoro:
    from mpas_tools.viz.paraview_extractor import extract_vtk
    extract_vtk(
            filename_pattern=ufile,
            variable_list='areaCell',
            dimension_list=['maxEdges=','nVertLevels=', 'nParticles='], 
            mesh_filename=ufile,
            out_dir=input_path, 
            ignore_time=True,
            # lonlat=True,
            xtime='none'
        )
    print("You could now visualise in Paraview (wireframe) the produced voronoi mesh!")
    print("This is a vtp mesh called: ", input_path+'/staticFieldsOnCells.vtp')



    
checkMesh = False

if checkMesh:
    import meshio

    paleovtk = input_path+"/init.vtk"

    vlist = list(primal_mesh.keys())
    vdata = []
    for k in vlist:
        vdata.append(primal_mesh[k].values)

    list_data = dict.fromkeys(el for el in vlist)
    list_data.update((k, vdata[i]) for i, k in enumerate(list_data))

    # Define mesh
    vis_mesh = meshio.Mesh(ucoords, {"triangle": ufaces}, 
                           point_data = list_data,
                        )
    # Write it disk
    meshio.write(paleovtk, vis_mesh)
    print("Writing VTK input file as {}".format(paleovtk))
    
    
    
    
    
meshname = var_path+"/mesh"
np.savez_compressed(meshname, v=ucoords, c=ufaces, 
                    z=dual_mesh.h.data
                    )


forcname = var_path+"/forcing250"

vel = np.zeros(ucoords.shape)
vel[:,0] = dual_mesh.vx.data
vel[:,1] = dual_mesh.vy.data
vel[:,2] = dual_mesh.vz.data
np.savez_compressed(forcname, 
                    vxyz=vel, 
                    #t=dual_mesh.tec.data, 
                    r=dual_mesh.rain.data,
                    #nz=dual_mesh.next_h.data,
                    )


widthCell = 50
highres = 25
lowres = 80
reso_contour = -50
input_path = "input_"+str(highres)+"_"+str(lowres)

# Build a background mesh with a 50 km width cell
ufcts.buildGlobalMeshSimple(widthCell, input_path)




# Loading the nc regular file
#ncgrid = xr.open_dataset('data/250.nc')[['h','rain']]

# Loading the UGRID file
ufile = input_path+'/mesh_'+str(widthCell)+'km.nc'
ugrid = uxr.open_grid(ufile) 

# Perform the interpolation (bilinear) 
var_path = 'vars_'+str(highres)+"_"+str(lowres)
var_name = 'coarse_250'
ufcts.inter2UGRID(ncgrid,ugrid,var_path,var_name,type='face')
data_file = [var_path+'/'+var_name+'.nc']

# Get the information related to the mesh: primal and dual mesh
primal_mesh = uxr.open_dataset(ufile, *data_file, use_dual=False)
dual_mesh = uxr.open_dataset(ufile, *data_file, use_dual=True)

# Extract nodes and faces information
ucoords = np.empty((dual_mesh.uxgrid.n_node,3))
ucoords[:,0] = dual_mesh.uxgrid.node_x.values
ucoords[:,1] = dual_mesh.uxgrid.node_y.values
ucoords[:,2] = dual_mesh.uxgrid.node_z.values
ufaces = primal_mesh.uxgrid.node_face_connectivity.values

# Get information about your mesh:
print("Number of nodes: ",len(ucoords)," | number of faces ",len(ufaces))
edge_min = np.round(dual_mesh.uxgrid.edge_node_distances.min().values/1000.+0.,2)
edge_max = np.round(dual_mesh.uxgrid.edge_node_distances.max().values/1000.+0.,2)
edge_mean = np.round(dual_mesh.uxgrid.edge_node_distances.mean().values/1000.+0.,2)
print("edge range (km): min ",edge_min," | max ",edge_max," | mean ",edge_mean)





vtkMesh = ufcts.generateVTKmesh(ucoords, ufaces)
dcoast = ufcts.distanceCoasts(vtkMesh, ucoords, dual_mesh.h.values, reso_contour)
ngrd = ufcts.getGridCoast(ncgrid, dual_mesh, dcoast, input_path)
ds, cellWidth = ufcts.cellWidthVsLatLonFuncDist(ngrd, width=[highres,lowres], maxdist=2.5e6)


"""
plotWeight = True
if plotWeight:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[8.0, 5.0])
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    im = ax.imshow(cellWidth, origin='lower',
                   vmin=cellWidth.min()-5,
                   vmax=cellWidth.max()+5,
                    transform=ccrs.PlateCarree(),
                    extent=[-180, 180, -90, 90], cmap='RdYlBu',
                    zorder=0)
    plt.title(
        'Grid cell size, km, min: {:.1f} max: {:.1f}'.format(
            cellWidth.min(),cellWidth.max()),fontsize=10)
    plt.colorbar(im, shrink=.60)
    fig.canvas.draw()
    plt.tight_layout()
    # plt.savefig(input_path+'/cellWidthGlobal.png', bbox_inches='tight')
    plt.show()
    plt.close()
"""

    
    
# Build the mesh
lon = ncgrid.lon.values
lat = ncgrid.lat.values
ufcts.refineGlobalMesh(cellWidth, lon, lat, input_path)




# Loading the UGRID file
ufile = input_path+'/mesh_refine.nc'
ugrid = uxr.open_grid(ufile) 

# Perform the interpolation (bilinear) 
var_path = 'vars_'+str(highres)+"_"+str(lowres)
var_name = 'refine_250'
ufcts.inter2UGRID(ncgrid,ugrid,var_path,var_name,type='face')
data_file = [var_path+'/'+var_name+'.nc']

# Get the information related to the mesh: primal and dual mesh
primal_mesh = uxr.open_dataset(ufile, *data_file, use_dual=False)
dual_mesh = uxr.open_dataset(ufile, *data_file, use_dual=True)

# Extract nodes and faces information
ucoords = np.empty((dual_mesh.uxgrid.n_node,3))
ucoords[:,0] = dual_mesh.uxgrid.node_x.values
ucoords[:,1] = dual_mesh.uxgrid.node_y.values
ucoords[:,2] = dual_mesh.uxgrid.node_z.values
ufaces = primal_mesh.uxgrid.node_face_connectivity.values

# Get information about your mesh:
print("Number of nodes: ",len(ucoords)," | number of faces ",len(ufaces))
edge_min = np.round(dual_mesh.uxgrid.edge_node_distances.min().values/1000.+0.,2)
edge_max = np.round(dual_mesh.uxgrid.edge_node_distances.max().values/1000.+0.,2)
edge_mean = np.round(dual_mesh.uxgrid.edge_node_distances.mean().values/1000.+0.,2)
print("edge range (km): min ",edge_min," | max ",edge_max," | mean ",edge_mean)





# Save voronoi mesh for visualisation purposes
saveVoro = True

if saveVoro:
    from mpas_tools.viz.paraview_extractor import extract_vtk
    extract_vtk(
            filename_pattern=ufile,
            variable_list='areaCell',
            dimension_list=['maxEdges=','nVertLevels=', 'nParticles='], 
            mesh_filename=ufile,
            out_dir=input_path, 
            ignore_time=True,
            # lonlat=True,
            xtime='none'
        )
    print("You could now visualise in Paraview (wireframe) the produced voronoi mesh!")
    print("This is a vtp mesh called: ", input_path+'/staticFieldsOnCells.vtp')




meshname = var_path+"/mesh"
np.savez_compressed(meshname, v=ucoords, c=ufaces, 
                    z=dual_mesh.h.data
                    )

forcname = var_path+"/rain250"
np.savez_compressed(forcname, 
                    r=dual_mesh.rain.data,
                    )


