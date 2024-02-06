# Import necessary libraries
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom


def plot_temperature_comparison(predicted, actual, step):
    plt.figure(figsize=(10, 5))
    plt.plot(predicted, label='Predicted')
    plt.plot(actual, label='Actual', linestyle='dashed')
    plt.xlabel('Day')
    plt.ylabel('Temperature (K)')
    plt.title(f'Predicted vs Actual Temperature for Los Angeles (Step {step})')
    plt.legend()
    plt.show()


# Load the NetCDF dataset
file_path = 'thist.nc'
dataset = nc.Dataset(file_path)

# Extract the necessary data
tas = dataset.variables['tas'][1,:,:]
lats = dataset.variables['lat'][:]
lons = dataset.variables['lon'][:]

# Define the coordinates for Los Angeles
la_coords = [(-118.25, 34.05)]  # Longitude, Latitude
LA_LAT = 34.05
LA_LON = 241.75  # Convert 118.25W to degrees east

# Find the grid points closest to Los Angeles
lat_idx = np.abs(lats - LA_LAT).argmin()
lon_idx = np.abs(lons - LA_LON).argmin()

# Create a map using Cartopy
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS)

# Highlight Los Angeles
la_point = sgeom.Point(la_coords)
ax.add_geometries([la_point.buffer(2)], ccrs.PlateCarree(), facecolor='none', edgecolor='red')

# Add the temperature data
plt.contourf(lons, lats, tas, 60, transform=ccrs.PlateCarree(), cmap='coolwarm')
plt.colorbar(label='Temperature (K)', orientation='vertical')
plt.title('Global Surface Air Temperature at Time Step [1]')
plt.show()
# Find the grid points closest to Los Angeles
lat_idx = np.abs(lats - 34.05).argmin()
lon_idx = np.abs(lons - 241.75).argmin()  # Convert 118.25W to degrees east

# Extract the temperature time series for this grid point
tas_la = dataset.variables['tas'][:, lat_idx, lon_idx]

# Plot the time series
times = dataset.variables['time'][:]
plt.figure(figsize=(12, 6))
plt.plot(times, tas_la)
plt.xlabel('Time')
plt.ylabel('Temperature (K)')
plt.title('Time Series of Temperature for Los Angeles')
plt.show()
