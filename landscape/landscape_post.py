import os
import pygmt
import pandas as pd
import xarray as xr
import numpy as np
from scripts import mapOutputs as mout

#import matplotlib.pyplot as plt
#%matplotlib inline



# Define output folder name for the simulation
out_path = 'export1'

if not os.path.exists(out_path):
    os.makedirs(out_path)
    


stp = 0

# Resolution of the netcdf structured grid
reso = 0.1

# Name of each netcdf output file
ncout = os.path.join(out_path, "data")

# Initialisation of the class
grid = mout.mapOutputs(path='./', filename='input-cont.yml', step=stp, uplift=False, flex=False)
    
  

    
for k in range(0,11):
    
    if stp>1:
        # Get goSPL variables
        grid.getData(stp)
        
    # Remap the variables on the regular mesh using distance weighting interpolation
    grid.buildLonLatMesh(res=reso, nghb=3)
    
    # Export corresponding regular mesh variables as netCDF file
    grid.exportNetCDF(ncfile = ncout+str(k)+'.nc')
    stp += 1
    
    
    
#dataset1 = xr.open_dataset(out_path+'/data1.nc')
#dataset5 = xr.open_dataset(out_path+'/data5.nc')
#dataset10 = xr.open_dataset(out_path+'/data10.nc')    
    
    
    
"""    
fig = pygmt.Figure()
# Plotting elevation
with pygmt.config(FONT='4p,Helvetica,black'):
    pygmt.makecpt(cmap="geo", series=[-6000, 6000])
    fig.basemap(region='d', projection='N12c', frame='afg')
    fig.grdimage(dataset1.elevation, shading='+a45+nt1', frame=False)
    # Add contour
    fig.grdcontour(
        interval=0.1,
        grid=dataset1.elevation,
        limit=[-0.1, 0.1],
    )
# Add color bar
with pygmt.config(FONT='5p,Helvetica,black'):    
    fig.colorbar(position="jBC+o0c/-1.35c+w6c/0.3c+h",frame=["a2000", "x+lElevation", "y+lm"])
# At time step
fig.text(text="Step 1", position="TL", font="8p,Helvetica-Bold,black") #, xshift="-0.75c")
fig.show(dpi=500, width=1000)

#####

fig = pygmt.Figure()
# Plotting elevation
with pygmt.config(FONT='4p,Helvetica,black'):
    pygmt.makecpt(cmap="geo", series=[-6000, 6000])
    fig.basemap(region='d', projection='N12c', frame='afg')
    fig.grdimage(dataset5.elevation, shading='+a45+nt1', frame=False)
    # Add contour
    fig.grdcontour(
        interval=0.1,
        grid=dataset5.elevation,
        limit=[-0.1, 0.1],
    )
# Add color bar
with pygmt.config(FONT='5p,Helvetica,black'):    
    fig.colorbar(position="jBC+o0c/-1.35c+w6c/0.3c+h",frame=["a2000", "x+lElevation", "y+lm"])
# At time step
fig.text(text="Step 5", position="TL", font="8p,Helvetica-Bold,black") #, xshift="-0.75c")
fig.show(dpi=500, width=1000)

#####

fig = pygmt.Figure()
# Plotting elevation
with pygmt.config(FONT='4p,Helvetica,black'):
    pygmt.makecpt(cmap="geo", series=[-6000, 6000])
    fig.basemap(region='d', projection='N12c', frame='afg')
    fig.grdimage(dataset10.elevation, shading='+a45+nt1', frame=False)
    # Add contour
    fig.grdcontour(
        interval=0.1,
        grid=dataset10.elevation,
        limit=[-0.1, 0.1],
    )
# Add color bar
with pygmt.config(FONT='5p,Helvetica,black'):    
    fig.colorbar(position="jBC+o0c/-1.35c+w6c/0.3c+h",frame=["a2000", "x+lElevation", "y+lm"])
# At time step
fig.text(text="Step 10", position="TL", font="8p,Helvetica-Bold,black") #, xshift="-0.75c")
fig.show(dpi=500, width=1000)
"""


"""
# Specify the folder containing the netcdf file from our simulation
out = 'export1/'

# Loop over each file and drop unwanted variables
for k in range(0,11,1):
    
    # Open the netcdf file
    dataset = xr.open_dataset(out+'data'+str(k)+'.nc')
    
    # Drop some variables (we only keep the sediment flow fluxes and the basin indices)
    reduce_ds = dataset[['flowDischarge','sedimentLoad','basinID']]
    
    # Save the reduced dataset as a new smaller netcdf
    reduce_ds.to_netcdf(out+'fsdata'+str(k)+'.nc')
"""    
    
    
#!mpirun -np 9 python3 getCatchmentInfo.py -i inputSedFlow.csv -o flowsed    
    
    
""" 
# Pick a time in Ma needs to be a integer
step = 10

# Get the fluxes files 
flowdf = pd.read_csv('flowsed/flow'+str(step)+'.csv')
seddf = pd.read_csv('flowsed/sed'+str(step)+'.csv')



logFA = np.log10(flowdf['val'].values)
logSed = np.log10(seddf['val'].values)



sorted_index_array = np.argsort(logFA)
sortedFA = logFA[sorted_index_array]
sortedLon = flowdf['lon'].values[sorted_index_array]
sortedLat = flowdf['lat'].values[sorted_index_array]

sorted_index_array1 = np.argsort(logSed)
sortedSed = logSed[sorted_index_array1]
sortedLon1 = seddf['lon'].values[sorted_index_array1]
sortedLat1 = seddf['lat'].values[sorted_index_array1]



# Define nth values
nlargest = 200

rLon = sortedLon[-nlargest : ]
rLat = sortedLat[-nlargest : ]
rFA = sortedFA[-nlargest : ]

rLon1 = sortedLon1[-nlargest : ]
rLat1 = sortedLat1[-nlargest : ]
rSed = sortedSed[-nlargest : ]
"""

