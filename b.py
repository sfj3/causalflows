import netCDF4 as nc
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
import seaborn as sns

def plot_fft_and_distribution_for_year(year_data, year, sample_rate=1):
    # Compute FFT of the year data
    fft_data = tf.signal.fft(tf.cast(year_data, dtype=tf.complex64))
    
    # Compute frequencies
    n = len(year_data)
    frequencies = np.fft.fftfreq(n, d=1./sample_rate)

    # Find peaks
    peaks, _ = find_peaks(np.abs(fft_data))

    # FFT Plot
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, np.abs(fft_data))
    plt.plot(frequencies[peaks], np.abs(fft_data)[peaks], "x")  # Mark the peaks
    plt.title(f'FFT of Precipitation Data for Year {year}')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()

    # Print peaks and their magnitudes
    print(f"Year {year}: Peaks at Frequencies and their Magnitudes")
    for peak in peaks:
        print(f"Frequency: {frequencies[peak]:.2f}, Magnitude: {np.abs(fft_data[peak]):.2f}")

    # Distribution Plot
    plt.figure(figsize=(12, 6))
    sns.histplot(year_data, kde=True)
    plt.title(f'Distribution of Precipitation Data for Year {year}')
    plt.xlabel('Precipitation')
    plt.ylabel('Frequency')
    plt.show()
    derivative_data = np.diff(year_data, n=1)

    # Plot the derivative
    plt.figure(figsize=(12, 6))
    plt.plot(derivative_data)
    plt.title(f'Derivative of Precipitation Data for Year {year}')
    plt.xlabel('Time (in pentads)')
    plt.ylabel('Change in Precipitation')
    plt.show()

    # Compute FFT of the derivative data
    fft_derivative = tf.signal.fft(tf.cast(derivative_data, dtype=tf.complex64))

    # Plot the FFT of the derivative
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies[:len(derivative_data)], np.abs(fft_derivative))
    plt.title(f'FFT of Derivative of Precipitation Data for Year {year}')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()

    # Plot the distribution of the original year data again for comparison
    plt.figure(figsize=(12, 6))
    sns.histplot(year_data, kde=True)
    plt.title(f'Distribution of Precipitation Data for Year {year}')
    plt.xlabel('Precipitation')
    plt.ylabel('Frequency')
    plt.show()


# using the distributions of the inputs (weights...) as different channels pass through
#batch norm prior, possibly inverse fft on freq data
# MADE Dense Layer for causal bias (how to implement causality using autocorrelation in time over temp pressure precip)
#for the made you can also implement causal flows wrt magnitudes (perform batchnorm prior and get scales) stdevs (same as prev) and frequencies (distributions)
#meta learning - applying gradient descent to hyperparameterization
#Since its bijective instead of reducing we will only include a single pt in loss and apply a mask for computational efficiency 
#should you use inverse FFT
#how to convolve magnitude with bijective functions over distributions most effectively
#should I consider wavelets 
#should you use MADE (masked dense layers)
#maybe you can characterize a year early in the pentad for better prediction?
#look at  year/year derivative(stdev) to see as well, if any or other yearly data helps... 
#look at multi year patterns
#look at selevtively using finer data to capture large scale effects 
#apply greater specificity to the predicted considering engineering relevance (ie summer month drought spread) -> must estimate distribution

#look at drivers of low temperature, number of days over a certain amount


#use temperature, sst

# Load NetCDF file
dataset = nc.Dataset('precip.pentad.mean.nc')

# Extract precipitation data
precip_data = dataset.variables['precip'][:]

# Reshape data to group into 12-month periods
# Assuming each year has 73 pentads (365 days ≈ 73 * 5 days)
# Adjust the number of pentads per year if necessary
pentads_per_year = 73
years = precip_data.shape[0] // pentads_per_year

# Reshape while handling incomplete years
reshaped_data = precip_data[:years * pentads_per_year].reshape(years, pentads_per_year, precip_data.shape[1], precip_data.shape[2])

# Convert to TensorFlow tensor
tf_data = tf.convert_to_tensor(reshaped_data, dtype=tf.float32)

# Assuming you want to work with TensorFlow Probability
# Create a distribution for the data
distribution = tfp.distributions.Normal(loc=tf.reduce_mean(tf_data), scale=tf.math.reduce_std(tf_data))
print(distribution)
# Now `distribution` contains the statistical model of your data
# Print a couple of examples
example_1 = tf_data[0, :2, :2, :2]  # First two pentads of the first year, first two latitudes and longitudes
example_2 = tf_data[1, :2, :2, :2]  # First two pentads of the second year, first two latitudes and longitudes

print("Example 1:", example_1.numpy())
print("Example 2:", example_2.numpy())

# Extract the time series for lat grid 22, lng grid 96
time_series_data = precip_data[:, 22, 96]

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(time_series_data)
plt.title('Time Series of Precipitation Data at Lat 22, Lng 96')
plt.xlabel('Time (in pentads)')
plt.ylabel('Precipitation')
plt.show()

# Calculate the distribution for this grid point
grid_point_distribution = tfp.distributions.Normal(
    loc=tf.reduce_mean(time_series_data),
    scale=tf.math.reduce_std(time_series_data)
)

# Plot the distribution
samples = grid_point_distribution.sample(1000).numpy()
plt.figure(figsize=(12, 6))
plt.hist(samples, bins=30, density=True)
plt.title('Distribution of Precipitation Data at Lat 22, Lng 96')
plt.xlabel('Precipitation')
plt.ylabel('Density')
plt.show()

# Compute FFT of the time series data
fft_data = tf.signal.fft(tf.cast(time_series_data, dtype=tf.complex64))

# Compute frequencies
sample_rate = 1  # Adjust if your data has a different sample rate
n = len(time_series_data)
frequencies = np.fft.fftfreq(n, d=1./sample_rate)

# Plot the FFT
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.abs(fft_data))
plt.title('FFT of Precipitation Data at LA Grid Point')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()

    # Example: Plotting FFT and printing peaks for the first two years
for year in range(2):  # Adjust the range for more years
    year_data = precip_data[year * pentads_per_year : (year + 1) * pentads_per_year, 22, 96]
    plot_fft_and_distribution_for_year(year_data, year + 1)  # Year number starts from 1 for readability

#create the distributions out of the frequency/magnitude plots then do bijective functions on them to transform them into predictions for the next year. 

