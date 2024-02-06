import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
import datetime
tfd = tfp.distributions
tfb = tfp.bijectors

# Ensure TensorFlow is executing eagerly
tf.config.run_functions_eagerly(True)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
summary_writer = tf.summary.create_file_writer(logdir)

# Define a Householder bijector
class Householder(tfb.Bijector):
    def __init__(self, vector, validate_args=False, name='householder'):
        self.vector = tf.Variable(vector / tf.norm(vector), trainable=True)
        super(Householder, self).__init__(
            forward_min_event_ndims=1, validate_args=validate_args, name=name)

    def _forward(self, x):
        v = self.vector
        # Adjusted to handle batch dimension correctly
        return x - 2 * tf.tensordot(x, v, axes=[[-1], [0]])[..., tf.newaxis] * v

    def _inverse(self, y):
        return self._forward(y)  # Reflection is its own inverse

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype)  # Jacobian determinant is 1


# Target distribution: 2D Gaussian with specified mean and covariance
mean = [2.0, 1.0]
covariance = [[2.0, 1.0], [1.0, 2.0]]
target_distribution = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=covariance)

tfd = tfp.distributions
tfb = tfp.bijectors

# Define the dimensions of your model
input_shape = [15552, 2]  # Adjust based on your actual input shape

# Masked Dense Flow
def masked_dense_flow(hidden_units, input_shape):
    flow_bijectors = []
    
    for units in hidden_units:
        flow_bijectors.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2, hidden_units=[units], input_order='right-to-left')))
        flow_bijectors.append(tfb.BatchNormalization())  # BatchNorm layer

    return tfb.Chain(flow_bijectors[::-1])

# Example usage
hidden_units = [512, 512]  # Define the number of hidden units in each MaskedDense layer
flow_bijector = masked_dense_flow(hidden_units, input_shape)

# Base distribution: standard normal
base_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(input_shape))

# Transformed distribution
flow_distribution = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=flow_bijector
)

# Training
num_steps = 1000
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        samples = base_distribution.sample(1024)
        loss = -tf.reduce_mean(flow_distribution.bijector.forward_log_det_jacobian(samples, event_ndims=1))
        loss += -tf.reduce_mean(flow_distribution.log_prob(samples))
    gradients = tape.gradient(loss, flow_distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, flow_distribution.trainable_variables))
    return loss
# Run the training loop with TensorBoard logging
for step in range(num_steps):
    loss = train_step()
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)
    if step % 100 == 0:
        print(f"Step: {step}, Loss: {loss}")

# TensorFlow Probability shorthands
tfd = tfp.distributions
tfb = tfp.bijectors

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Setup for TensorBoard logging
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
summary_writer = tf.summary.create_file_writer(logdir)

# Define the dimensions of your model
input_shape = [15552, 2]  # Adjust based on your actual input shape

# Masked Dense Flow
def masked_dense_flow(hidden_units):
    flow_bijectors = []
    for units in hidden_units:
        flow_bijectors.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2, hidden_units=[units], input_order='right-to-left')))
        flow_bijectors.append(tfb.BatchNormalization())
    return tfb.Chain(flow_bijectors[::-1])

# Define the number of hidden units in each Masked Dense layer
hidden_units = [512, 512]

# Create the flow bijector
flow_bijector = masked_dense_flow(hidden_units)

# Base distribution: standard normal
base_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros([np.prod(input_shape)]))

# Transformed distribution
flow_distribution = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=flow_bijector
)

# Training configurations
num_steps = 1000
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# Training function
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        samples = base_distribution.sample(1024)
        loss = -tf.reduce_mean(flow_distribution.log_prob(samples))
    gradients = tape.gradient(loss, flow_distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, flow_distribution.trainable_variables))
    return loss

# Training loop
for step in range(num_steps):
    loss = train_step()
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)
    if step % 100 == 0:
        print(f"Step: {step}, Loss: {loss}")
