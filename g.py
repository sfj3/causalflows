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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages are logged, 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers



file_path = 'thist.nc'
dataset = nc.Dataset(file_path)



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

    DAYS = 365

    model_inputs = []
    actual_data = []
    total_periods = global_temperature_data.shape[0] - DAYS - days_to_estimate + 1

    for start_day in range(0, total_periods, days_to_estimate):
        # Extract the 365-day window for FFT
        data_window = global_temperature_data[start_day:start_day + DAYS]

        # Perform FFT along the time dimension and get amplitude and phase


        fft_data = np.fft.fft(data_window, axis=0)


        # Combine flattened amplitudes and phases
        model_inputs.append(np.array([fft_data[0],fft_data[1]]))
        la_data = global_temperature_data[start_day + DAYS:start_day + DAYS + days_to_estimate, lat_idx, lon_idx]
        actual_data.append(la_data.flatten())



    return model_inputs[1:], np.array(actual_data[1:])


model_inputs,actual_data = create_training_data(dataset)


class FFModel(tf.keras.Model):
    def __init__(self, num_transformations=1):
        super(FFModel, self).__init__()
        # Layers for processing
        self.batch_up = tf.keras.layers.BatchNormalization()
        self.batch_low = tf.keras.layers.BatchNormalization()
        self.dense_up = tf.keras.layers.Dense(90*144, activation='relu')
        self.dense_low = tf.keras.layers.Dense(90*144, activation='relu')
        self.dense_up_2 = tf.keras.layers.Dense(90*144, activation='relu')
        self.dense_low_2 = tf.keras.layers.Dense(90*144, activation='relu')
        self.num_transformations = num_transformations
        self.softmax = tf.keras.layers.Softmax()
        # Shift and scale parameters
        self.c = tf.keras.Sequential([
            tf.keras.layers.Reshape((90, 144, 1), input_shape=(90, 144)),  # Add channel dimension
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Reshape((10, 10))  # Reshape to desired output
        ])
        self.d = tf.keras.Sequential([
            tf.keras.layers.Reshape((90, 144, 1), input_shape=(90, 144)),  # Add channel dimension
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Reshape((10, 10))  # Reshape to desired output
        ])
        self.shift = tf.Variable(initial_value=4*tf.ones(365), trainable=True, dtype=tf.float32)
        self.scale = tf.Variable(initial_value=1.1*tf.ones(365),trainable=True,dtype=tf.float32)
    def call(self, inputs):

        test_input = inputs
        up = tf.reshape(self.c(tf.reshape(tf.math.abs(test_input[0]),[1,90,144])),[1,100])
        low = tf.reshape(self.d(tf.reshape(tf.math.abs(test_input[-1]),[1,90,144])),[1,100])
        normalized_low = low# (low - tf.reduce_min(low)) / (tf.reduce_max(low) - tf.reduce_min(low))
        normalized_up = up#(up - tf.reduce_min(up)) / (tf.reduce_max(up) - tf.reduce_min(up))
        
        low_1 = normalized_low#self.batch_low(normalized_low)              check to see why batch norm wasnt working there
        up_1 = normalized_up#self.batch_up(normalized_up)
        
        up_1 = self.dense_up(up_1)
        up_1 = self.dense_up_2(up_1)
        low_1 = self.dense_low(low_1)
        low_1 = self.dense_low_2(low_1)
        up_1 = self.softmax(up_1)
        low_1 = self.softmax(low_1)
        up_1 = tf.reshape(up_1,[90,144]) #these dont seem to be learning, maybe the problem is softmax isnt changing the probs 
        low_1 = tf.reshape(low_1,[90,144])
        up_weighted = tf.cast(up_1,tf.complex64) * tf.cast(test_input[0]**2,tf.complex64)
        low_weighted = tf.cast(low_1,tf.complex64) * tf.cast(test_input[1]**2,tf.complex64)
        up_weight = tf.reduce_sum(up_weighted)
        low_weight = tf.reduce_sum(low_weighted)
        values = tf.concat([[up_weight], tf.fill([363], tf.constant(20, dtype=tf.complex64)), [low_weight]], 0)#try other values for the constant here
        inv_vals = tf.sqrt(tf.math.real(tf.signal.ifft(values))/365)
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
        transformed_samples = []
        
        transformed_sample = (samples) * self.scale + self.shift
        transformed_samples.append(transformed_sample)

        return transformed_samples

# Assuming create_training_data is defined
model_inputs, actual_data = create_training_data(dataset)

# Instantiate and compile the model
model = FFModel(num_transformations=1)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.005)#add a learning rate schedule
loss_fn = tf.keras.losses.MeanSquaredError()
i = 0
for epoch in range(100):  # For each epoch
    for inputs, targets in zip(model_inputs, actual_data):  # Iterate through each set
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)  # Forward pass
            loss = loss_fn(tf.sort(targets,axis=0), tf.sort(predictions[0],axis=0)) + tf.square(tf.square(tf.cast(tf.math.reduce_sum(tf.cast(targets < 273.15, tf.float32)), tf.float32) - tf.cast(tf.math.reduce_sum(tf.cast(predictions[0] < 273.15, tf.float32)), tf.float32)))
            print('loss',loss,'step',i,'days',tf.square(tf.square(tf.cast(tf.math.reduce_sum(tf.cast(targets < 273.15, tf.float32)), tf.float32) - tf.cast(tf.math.reduce_sum(tf.cast(predictions[0] < 273.15, tf.float32)), tf.float32))))
            i = i + 1
        gradients = tape.gradient(loss, model.trainable_variables)  # Compute gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Update weights


#use log prob or something else.., play around with different loss functions
# play with learning rate add a schedule
#play with convolutional and imaginary convolutional layers 
#play with distribution
#add some type of other state classifier or a transformer... 