import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import reparameterize, kernel_rbf

class Layer_rbf():
    def __init__(self, input_dim, output_dim, num_inducings, mean_function=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducings = num_inducings

        self.Z = np.zeros((self.num_inducings, self.input_dim))
        self.q_mu = np.zeros((self.num_inducings, self.output_dim))
        self.q_sqrt = np.vstack([np.expand_dims(np.eye(self.num_inducings), 0) \
                                                for _ in range(self.output_dim)])
        self.theta = np.zeros((self.input_dim+1, 1))
        self.needs_build_cholesky = True
        self.mean_function = mean_function
        self.A = None
        self.noise = tf.exp(tf.Variable(tf.log(1e-4), dtype=tf.float32))
        # self.b = tf.Variable(1.0, dtype=tf.float32)

    def find_weights(self, input_dim, output_dim, X):
        if input_dim == output_dim:
            W = np.eye(input_dim)
        elif input_dim > output_dim:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            W = V[:output_dim, :].T
        else:
            I = np.eye(input_dim)
            zeros  = np.zeros((input_dim, output_dim - input_dim))
            W = np.concatenate([I, zeros], 1)
        return W

    def compute_inputs(self, X, Z):
        W = self.find_weights(self.input_dim, self.output_dim, X)
        Z_running = Z.copy().dot(W)
        X_running = X.copy().dot(W)
        return X_running, Z_running, W

    def initialize_forward(self, X, Z):
        X_running, Z_running, W = self.compute_inputs(X, Z)
        self.Z = tf.Variable(Z, dtype=tf.float32)
        self.q_mu = tf.Variable(self.q_mu, dtype=tf.float32)

        Ku = kernel_rbf(self.theta, Z, Z)
        Lu = tf.to_float(tf.cholesky(Ku + self.noise * tf.eye(Z.shape[0], dtype=tf.float32)))
        logLu = tf.log(Lu)
        self.q_sqrt = tf.exp(tf.Variable(tf.tile(logLu[None,:,:], [self.output_dim, 1, 1]), dtype=tf.float32))
        self.theta = tf.exp(tf.Variable(self.theta, dtype=tf.float32))
        if self.mean_function is not None:
            self.A = tf.Variable(W, dtype=tf.float32)
        return X_running, Z_running

    def  build_cholesky_if_needed(self):
        if self.needs_build_cholesky:
            self.Ku = kernel_rbf(self.theta, self.Z, self.Z) + self.noise * tf.eye(self.num_inducings)
            self.Lu = tf.cholesky(self.Ku)
            self.Ku_tiled = tf.tile(self.Ku[None, :, :], [self.output_dim, 1, 1])
            self.Lu_tiled = tf.tile(self.Lu[None, :, :], [self.output_dim, 1, 1])
            self.needs_build_cholesky = False

    def conditional_ND(self, X, full_cov=False):
        self.build_cholesky_if_needed()

        Kuf = kernel_rbf(self.theta, self.Z, X)
        A = tf.matrix_triangular_solve(self.Lu, Kuf, lower=True)
        A = tf.matrix_triangular_solve(tf.transpose(self.Lu), A, lower=False)
        mean = tf.matmul(A, self.q_mu, transpose_a=True)

        A_tiled = tf.tile(A[None, :, :], [self.output_dim, 1, 1])
        SK = -self.Ku_tiled

        if self.q_sqrt is not None:
            SK += tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True)
        B = tf.matmul(SK, A_tiled)

        if full_cov:
            # (num_latent, num_X, num_X)
            delta_cov = tf.matmul(A_tiled, B, transpose_a=True)
            Kff = kernel_rbf(self.theta, X, X)
        else:
            # (num_latent, num_X)
            delta_cov = tf.reduce_sum(A_tiled * B, 1)
            Kff = tf.diag_part(kernel_rbf(self.theta, X, X))

        var = tf.expand_dims(Kff, 0) + delta_cov
        var = tf.transpose(var)

        if self.A is not None:
            # return mean + tf.matmul(tf.cast(X, dtype=tf.float32), self.A) + self.b, var
            return mean + tf.matmul(tf.cast(X, dtype=tf.float32), self.A), var
        else:
            return mean, var
        # return mean + self.mean_function(X), var

    def conditional_SND(self, X, full_cov=False):
        if full_cov is True:
            f = lambda a: self.conditional_ND(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float32, tf.float32))
            return tf.stack(mean), tf.stack(var)
        else:
            S, D = X.shape[0], X.shape[2]
            X_flat = tf.reshape(X, [-1, D])
            mean, var = self.conditional_ND(X_flat)

            return [tf.reshape(m, [S, -1, self.output_dim]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):

        mean, var = self.conditional_SND(X, full_cov=full_cov)

        # set shapes
        S = X.shape[0]
        D = self.output_dim
        mean = tf.reshape(mean, (S, -1, D))
        if full_cov:
            var = tf.reshape(var, (S, N, N, D))
        else:
            var = tf.reshape(var, (S, -1, D))

        if z is None:
            z = tf.random_normal(tf.shape(mean), dtype=tf.float32)
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        return samples, mean, var

    def KL(self):

        self.build_cholesky_if_needed()

        KL = -0.5 * self.output_dim * self.num_inducings
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.q_sqrt) ** 2))

        KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.Lu))) * self.output_dim
        KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.Lu_tiled, self.q_sqrt, lower=True)))
        Kinv_m = tf.cholesky_solve(self.Lu, self.q_mu)
        KL += 0.5 * tf.reduce_sum(self.q_mu * Kinv_m)

        return KL