import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    # Open the dataset and return the variable
    dataset = nc.Dataset(file_name)
    return dataset.variables

# Load the anomaly and long-term mean datasets
anomaly_data = load_data("air.mon.anom.median.nc")['air']
ltm_data = load_data("air.mon.ltm.nc")['air']

# Initialize an array to store the actual temperature data
actual_temp = np.zeros(anomaly_data.shape)

# Process each time step
for t in range(anomaly_data.shape[0]):
    print(t)
    for lat in range(anomaly_data.shape[1]):
        for lon in range(anomaly_data.shape[2]):
            # Check if the data is a number and handle missing data
            if np.isnan(anomaly_data[t, lat, lon]) or np.isnan(ltm_data[t % ltm_data.shape[0], lat, lon]):
                actual_temp[t, lat, lon] = np.nan
            else:
                actual_temp[t, lat, lon] = anomaly_data[t, lat, lon] + ltm_data[t % ltm_data.shape[0], lat, lon]

# Save the actual temperature data
np.save("actual_temperatures.npy", actual_temp)

# Plot a time series for a specific lat and lon as a check
lat_idx = 0  # example latitude index
lon_idx = 0  # example longitude index
plt.plot(actual_temp[:, lat_idx, lon_idx])
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Time Series at a Specific Grid Point')
plt.show()
