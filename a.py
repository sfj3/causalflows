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

# Base distribution: Standard normal
base_distribution = tfd.MultivariateNormalDiag(loc=[0., 0.])

# Normalizing Flow: Chain of bijectors
flow_bijectors = [

    Householder(vector=[1., 0.]),
    tfb.Permute(permutation=[1, 0]),
    tfb.BatchNormalization(),
    Householder(vector=[0., 1.]),
    tfb.BatchNormalization(),
    Householder(vector=[1., 0.]),
    tfb.Permute(permutation=[1, 0]),
    tfb.BatchNormalization(),
    Householder(vector=[0., 1.]),
    # Add more bijectors as needed
]

normalizing_flow_bijector = tfb.Chain(flow_bijectors[::-1])

# Transformed distribution
flow_distribution = tfd.TransformedDistribution(distribution=base_distribution, bijector=normalizing_flow_bijector)

# Training
num_steps = 1000
optimizer = tf.optimizers.Adam(learning_rate=0.001)

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

# Base distribution: Standard normal
base_distribution = tfd.MultivariateNormalDiag(loc=[0., 0.])

# Normalizing Flow: Chain of bijectors
flow_bijectors = [

    Householder(vector=[1., 0.]),
    tfb.Permute(permutation=[1, 0]),
    tfb.BatchNormalization(),
    Householder(vector=[0., 1.]),
    tfb.BatchNormalization(),
    Householder(vector=[1., 0.]),
    tfb.Permute(permutation=[1, 0]),
    tfb.BatchNormalization(),
    Householder(vector=[0., 1.]),
    # Add more bijectors as needed
]

normalizing_flow_bijector = tfb.Chain(flow_bijectors[::-1])

# Transformed distribution
flow_distribution = tfd.TransformedDistribution(distribution=base_distribution, bijector=normalizing_flow_bijector)

# Training
num_steps = 1000
optimizer = tf.optimizers.Adam(learning_rate=0.001)

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

