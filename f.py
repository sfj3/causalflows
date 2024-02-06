import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import netCDF4 as nc
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
# Suppress TensorFlow logging except for errors
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages are logged, 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed

# Load dataset
file_path = 'thist.nc'
dataset = nc.Dataset(file_path)



# Extract the necessary data
tas = dataset.variables['tas'][1,:,:]
lats = dataset.variables['lat'][:]

lons = dataset.variables['lon'][:]
lat_idx = np.abs(lats - 34.05).argmin()
lon_idx = np.abs(lons - 241.75).argmin()  # Convert 118.25W to degrees east
# Constants and configurations
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DAYS = 365
FFT_LENGTH = DAYS // 2 + 1



def create_training_data(dataset, days_to_estimate=365):
    global_temperature_data = dataset.variables['tas'][:]

    # Constants
    DAYS = 365

    # Preallocate arrays
    model_inputs = []
    actual_data = []
    conv_inputs = []
    # Determine the number of periods to generate data for
    total_periods = global_temperature_data.shape[0] - DAYS - days_to_estimate + 1

    for start_day in range(0, total_periods, days_to_estimate):
        # Extract the 365-day window for FFT
        data_window = global_temperature_data[start_day:start_day + DAYS]

        # Perform FFT along the time dimension and get amplitude and phase
        # data_window = np.random.rand(1024)  # Example data

        fft_data = np.fft.fft(data_window, axis=0)


        # Combine flattened amplitudes and phases
        model_inputs.append(np.array([fft_data[0],fft_data[1]]))
        la_data = global_temperature_data[start_day + DAYS:start_day + DAYS + days_to_estimate, lat_idx, lon_idx]
        actual_data.append(la_data.flatten())



    return model_inputs[1:], np.array(actual_data[1:])#use weighted average here. 


model_inputs,actual_data = create_training_data(dataset)

print('start training')
test_input = model_inputs[0] #example of one year of gridded temperature data, you can batch this
test_actual = actual_data[0]
up = tf.math.abs(tf.reshape(test_input[0],[1,90*144])) #the two peaks of the fft
low = tf.math.abs(tf.reshape(test_input[1],[1,90*144]))
normalized_low = (low - tf.reduce_min(low)) / (tf.reduce_max(low) - tf.reduce_min(low))
normalized_up = (up - tf.reduce_min(up)) / (tf.reduce_max(up) - tf.reduce_min(up))
batch_up = tf.keras.layers.BatchNormalization()
batch_low = tf.keras.layers.BatchNormalization()
low_1 = batch_low(normalized_low)
up_1 = batch_up(normalized_up)
dense_low = tf.keras.layers.Dense(90*144)
dense_up = tf.keras.layers.Dense(90*144)
up_1 = dense_up(up_1)
low_1 = dense_low(low_1)
softmax = tf.keras.layers.Softmax()
up_1 = softmax(up_1)
low_1 = softmax(low_1)
up_1 = tf.reshape(up_1,[90,144])
low_1 = tf.reshape(low_1,[90,144])
up_weighted = tf.cast(up_1,tf.complex64) * test_input[0]
low_weighted = tf.cast(low_1,tf.complex64) * test_input[1]
up_weight = tf.reduce_sum(up_weighted)
low_weight = tf.reduce_sum(low_weighted)
values = tf.concat([[up_weight], tf.fill([363], tf.constant(30, dtype=tf.complex64)), [low_weight]], 0)
inv_vals = tf.math.real(tf.signal.ifft(values))
# Compute the statistics from `inv_vals`
min_val = tf.reduce_min(inv_vals[1:])#the first value is kinda strange
max_val = tf.reduce_max(inv_vals[1:])
mean_val = tf.reduce_mean(inv_vals[1:])
std_dev_val = tf.math.reduce_std(inv_vals[1:])

# Define the truncated normal distribution
truncated_normal_dist = tfp.distributions.TruncatedNormal(loc=mean_val,
                                                          scale=std_dev_val,
                                                          low=min_val,
                                                          high=max_val)

# Sample from the distribution
samples = truncated_normal_dist.sample(365)
#now perform shift and scale on this distribution, learning a the shift and the scale tensors
#samples = shift_and_scale(samples)
#finally evaluate the model performance
mse = tf.reduce_mean(tf.square(tf.sort(samples) - tf.sort(actual_data[0]))).np()

#the model should learn on the shift and scale and the dense. 

