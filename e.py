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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages are logged, 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed

# Additional setup to ensure TensorFlow does not reinitialize its logging system afterwards
tf.get_logger().setLevel('ERROR')

def normalize_grid(tensor):
    # Assume tensor shape is [batch_size, height, width, channels] = [30, 90, 144, 1]
    shape = tf.shape(tensor)
    batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    
    # Flatten the height and width dimensions
    tensor_flat = tf.reshape(tensor, [batch_size, height * width * channels])
    
    # Apply softmax to normalize each grid so that it sums to 1
    tensor_normalized = tf.nn.softmax(tensor_flat, axis=-1)
    
    # Reshape back to the original shape
    tensor_final = tf.reshape(tensor_normalized, [batch_size, height, width, channels])
    
    return tensor_final

cnn_model = Sequential([
    # First layer: Conv2D with 'same' padding to keep dimensions unchanged
    Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(90, 144, 1)),
    
    # Middle layer: Conv2D with dilation and 'same' padding
    Conv2D(filters=32, kernel_size=(3, 3), padding='same', dilation_rate=2, activation='relu'),
    
    # Third layer: Conv2D to ensure output matches input dimensions
    Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')
])
cnn_model.compile(optimizer='adam', loss='mean_squared_error')

def plot_distribution_comparison(predicted, actual, step):
    plt.figure(figsize=(10, 5))
    sns.histplot(predicted, color="blue", label='Predicted', kde=True, stat="density", linewidth=0)
    sns.histplot(actual, color="red", label='Actual', kde=True, stat="density", linewidth=0)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Density')
    plt.title(f'Distribution of Predicted vs Actual Temperature for Los Angeles (Step {step})')
    plt.legend()
    plt.show()

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

        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(np.abs(fft_data[:][0][0]))
        plt.title('FFT Magnitude')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()
        conv_inputs.append(2)

        # Combine flattened amplitudes and phases
        model_inputs.append(fft_data)
        la_data = global_temperature_data[start_day + DAYS:start_day + DAYS + days_to_estimate, lat_idx, lon_idx]
        actual_data.append(la_data.flatten())



    return np.average(np.reshape(np.array(conv_inputs),(30,365,90,144,1)),axis=1),tf.transpose(np.array(model_inputs),[0,2,3,1]), np.array(actual_data)#use weighted average here. 

    # return [np.abs(np.fft.ifft(np.average(np.array(model_inputs)[i,:],axis=(1,2)))) for i in range(len(model_inputs))], np.array(actual_data)#use tensorflow weighted average here. 






conv_inputs,model_inputs, actual_data = create_training_data(dataset)
print('do')



##++++++++++________+++++++++++_________++++++++++++++++___________+++++
# TensorFlow Probability components


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfb = tfp.bijectors
tfd = tfp.distributions

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfb = tfp.bijectors
tfd = tfp.distributions

class AutoregressiveModel(tfk.Model):
    def __init__(self, input_shape, hidden_units=[2, 4, 8, 16, 32, 64, 64, 64]):
        super(AutoregressiveModel, self).__init__()
        self.event_size = np.prod(input_shape)
        
        def distribution_fn(samples):
            params = self.made(samples)
            means, log_scales, logit = tf.split(params, [2, 2, 1], axis=-1)
            scales = tf.nn.softplus(log_scales)
            logits = tf.repeat(logit, repeats=samples.shape[0], axis=0)

            mixture_distribution = tfd.Categorical(logits=logits)
            components_distribution = tfd.Normal(loc=means, scale=scales)
            
            return tfd.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                components_distribution=tfd.Independent(components_distribution, reinterpreted_batch_ndims=1)
            )

        self.made = tfb.AutoregressiveNetwork(
            params=5,  # For two means, two variances, and one mixture coefficient
            hidden_units=hidden_units,
            event_shape=[self.event_size],
            conditional=False
        )

        # This is where the autoregressive_distribution should be assigned
        self.autoregressive_distribution = tfd.Autoregressive(
            distribution_fn=distribution_fn,
            sample0=tf.zeros([self.event_size], dtype=tf.float32),
            num_steps=None,
            validate_args=False,
            allow_nan_stats=True,
            name='Autoregressive'
        )

    def call(self, x):
        x_flat = tf.reshape(x, [-1, self.event_size])
        log_prob = self.autoregressive_distribution.log_prob(x_flat)
        return log_prob

    def sample(self, sample_shape):
        return self.autoregressive_distribution.sample(sample_shape)





class ConditionalMADE(tfk.Model):
    def __init__(self, input_shape, conditional_shape, hidden_units=[2, 2]):
        super(ConditionalMADE, self).__init__()
        self.conditional_shape = conditional_shape
        
        # Initialize min_val and max_val as learnable parameters, these should actually be functions of the input though
        self.min_val = tf.Variable(250, name="min_val", dtype=tf.float32, trainable=True)
        self.max_val = tf.Variable(350, name="max_val", dtype=tf.float32, trainable=True)
        
        self.made = tfb.AutoregressiveNetwork(
            params=2,
            hidden_units=hidden_units,
            event_shape=input_shape,
            conditional=True,
            conditional_event_shape=(np.prod(conditional_shape),)
        )

    def build_distribution(self, params):
        mean, log_scale = tf.split(params, 2, axis=-1)
        scale = tf.nn.softplus(tf.tanh(log_scale))  # Ensure scale is positive
        
        # Use the learnable min_val and max_val for the TruncatedNormal distribution
        return tfd.TruncatedNormal(loc=mean, scale=scale, low=self.min_val, high=self.max_val)

    def call(self, x, conditional_input):
        params = self.made(x, conditional_input=conditional_input)
        dist = self.build_distribution(params)
        return dist.log_prob(tf.reshape(x, [-1, 365, 1]))
    def sample(self, sample_shape, conditional_input):
        # Create a dummy input tensor for 'x'. The shape must match what 'self.made' expects.
        # Here, I'm assuming 'self.made' expects a tensor of shape [batch_size, input_shape], but you'll need to adjust this based on your model's specifics.
        dummy_input_shape = [1] + list(self.conditional_shape)  # Adjust based on the expected input shape of 'self.made'
        dummy_input = tf.zeros(dummy_input_shape)

        # Now, call 'self.made' with the dummy input as the first argument and 'conditional_input' as a named argument.
        params = self.made(dummy_input, conditional_input=conditional_input)

        # Continue as before to build the distribution and sample from it.
        dist = self.build_distribution(params)
        return dist.sample(sample_shape)

    
    
    
    
    
    
    
    # def call(self, x, conditional_input, min_val, max_val):
    #     params = self.made(x, conditional_input=conditional_input)
    #     dist = self.build_distribution(params, min_val=min_val, max_val=max_val)
    #     return dist.log_prob(tf.reshape(x, [-1, 365, 1]))

    
    
    
    
    
    
    
    
    
    # def sample(self, sample_shape, conditional_input, min_val, max_val):
    #     # Create a dummy tensor with the appropriate shape for the autoregressive network
    #     # This shape should match the expected input shape for self.made
    #     dummy_input = tf.zeros_like(conditional_input)  # Adjust this based on the actual expected shape

    #     # Generate parameters using the dummy input and the conditional input
    #     params = self.made(dummy_input, conditional_input=conditional_input)

    #     # Now proceed to build and sample from the distribution as before
    #     dist = self.build_distribution(params, min_val=min_val, max_val=max_val)
        
    #     return dist.sample(sample_shape)



# Adjust the made_model initialization as per the new signature
made_model = AutoregressiveModel(input_shape=(365,))


# Custom training step
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
# optimizer1 = tf.keras.optimizers.legacy.Adam(learning_rate=0.1)
#@tf.function
def train_step(conv_inputs, model_inputs, actual_data):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass through CNN to obtain weights
        # weights = normalize_grid(cnn_model(conv_inputs))
        # weights_complex = tf.cast(weights, tf.dtypes.complex128)
        # weights_expanded = tf.broadcast_to(weights_complex, tf.shape(model_inputs))
        # weighted_inputs = model_inputs * weights_expanded
        # reduced_inputs = tf.reduce_sum(weighted_inputs, axis=[1, 2])
        # reduced_inputs = tf.signal.ifft(reduced_inputs)
        # made_conditional = tf.cast(tf.math.abs(reduced_inputs), tf.float32)
        
        # Dynamically determine min and max values from the CNN model outputs or another source
        # For this example, let's assume these are fixed or derived from another part of your model
        # min_val = tf.reduce_min(made_conditional)  # Placeholder for dynamic min value, change this next

        # max_val = tf.reduce_max(made_conditional)  # Placeholder for dynamic max value
        

        estimate = tf.reshape(made_model.sample((1,)), (365,))
        actual = tf.reshape(actual_data, (365,))
        below_estimate = tf.reduce_sum(tf.cast(estimate < 273.15, tf.float32))
        below_actual = tf.reduce_sum(tf.cast(actual < 273.15, tf.float32))
        log_prob = made_model(actual_data)
        log_prob = tf.where(tf.math.is_inf(log_prob), tf.fill(tf.shape(log_prob), -5000.0), log_prob)
        loss_diff = tf.abs(below_estimate - below_actual)
        loss =  - tf.reduce_sum(log_prob)
        # regularization_penalty = tf.reduce_mean(tf.square(made_model.min_val - np.min(actual_data)) + tf.square(made_model.max_val - np.max(actual_data)))
        # loss = loss_diff#loss + regularization_penalty **2
    # Compute gradients
    # gradients_cnn = tape.gradient(loss, cnn_model.trainable_variables)
    gradients_made = tape.gradient(loss, made_model.trainable_variables)

    # Apply gradients if they are not None
    # min_val_gradient, max_val_gradient = tape.gradient(loss, [made_model.min_val, made_model.max_val])
    # optimizer1.apply_gradients(zip([min_val_gradient * 1000, max_val_gradient * 1000], [made_model.min_val, made_model.max_val]))
    # gradients_cnn = [(grad, var) for grad, var in zip(gradients_cnn, cnn_model.trainable_variables) if grad is not None]
    gradients_made = [(grad, var) for grad, var in zip(gradients_made, made_model.trainable_variables) if grad is not None]
    # if gradients_cnn:
    #     optimizer.apply_gradients(gradients_cnn)
    if gradients_made:
        optimizer.apply_gradients(gradients_made)
    return loss


# Training loop
EPOCHS = 29000
BATCH_SIZE = 1

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for batch in range(0, len(model_inputs), BATCH_SIZE):
        conv_input_batch = conv_inputs[batch:batch+BATCH_SIZE]
        model_input_batch = model_inputs[batch:batch+BATCH_SIZE]
        actual_data_batch = actual_data[batch:batch+BATCH_SIZE]
        loss = train_step(conv_input_batch, model_input_batch, actual_data_batch)#add to this the key distributional point you want to have 
        print(f"Batch {batch//BATCH_SIZE+1}, Loss: {loss.numpy()}")