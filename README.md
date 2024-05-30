Repository containing code to generate datasets for a synthetic world based on a GPlates-format topological reconstruction model. Primarily created for teaching purposes. 

# Steps for use

1. Define a reconstructon model in GPlates - this should comprise a rotation model, polygons defining the continents, and topological plate boundaries
2. Optionally define your own Geomagnetic Polarity Timescale, hotspot locations to generate seamount trails
3. Open the main notebook and specify the files for your reconstruction model near the top
4. Run the notebook to generate datasets

# Outputs
1. Grids of seafloor age
2. Grids of elevation and bathymetry. Elevations include mountain ranges for parts of continents adjacent to subduction zones or undergoing continent-continent collision
3. Grids of Total Magnetic Intensity (with 'magnetic stripes' corresponding to the seafloor spreading history
4. Point distributions approximating earthquakes along plate boundaries and subducting slabs at depth
5. Point data approximating aspects of the geological record such as climatically sensitive lithologies (ie confined to latitude bands) or volcanism along subduction zones

