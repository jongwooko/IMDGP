from utils import kernel_rbf
from layer import Layer_rbf
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from dsdgp import DSDGP

class LOWDGP():
#     def __init__(self, x_tr, y_tr, layers, num_samples=20):
#         self.x_tr, self.y_tr, self.layers = x_tr.astype(np.float32), y_tr.astype(np.float32), layers
#         self.f_tr = np.concatenate((np.ones((self.x_tr.shape[0], 1)), self.x_tr), axis=1).astype(np.float32)
#         self.num_samples = num_samples
#         self.input_dim, self.output_dim = x_tr.shape[1], y_tr.shape[1]
#         self.num_data = x_tr.shape[0]
#         self.prior_m_beta = np.zeros((self.input_dim + 1, 1), dtype=np.float32)
#         self.prior_s_beta = np.ones((self.input_dim + 1), dtype=np.float32)
#         self.m_beta = tf.Variable(np.zeros([self.input_dim + 1, 1]), dtype=tf.float32)
#         self.s_beta = tf.exp(tf.Variable(np.zeros([self.input_dim + 1, ]), dtype=tf.float32))
#         self.variance = tf.exp(tf.Variable(tf.log(1e-4), dtype=tf.float32))

#         self.dsdgp = DSDGP(X=self.x_tr, Y=None, layers=self.layers, Z=None, num_samples=self.num_samples)
#         self.dsdgp.initialize_param()
    
    def __init__(self, num_data, input_dim, output_dim, layers, num_samples=20):
        self.num_data, self.num_samples = num_data, num_samples
        self.input_dim, self.output_dim = input_dim, output_dim
        self.prior_m_beta = np.zeros((self.input_dim + 1, 1), dtype=np.float32)
        self.prior_s_beta = np.ones((self.input_dim + 1), dtype=np.float32)
        self.m_beta = tf.Variable(np.zeros([self.input_dim + 1, 1]), dtype=tf.float32)
        self.s_beta = tf.exp(tf.Variable(np.zeros([self.input_dim + 1,]), dtype=tf.float32))
        self.variance = tf.exp(tf.Variable(tf.log(1e-4), dtype=tf.float32))
        
        self.dsdgp = DSDGP(num_data=num_data, input_dim=input_dim, output_dim=output_dim, \
                           layers=layers, num_samples=num_samples)
        self.dsdgp.initialize_param()

#     def _build_likelihood(self):
#         KL_gamma = tf.reduce_sum([layer.KL() for layer in self.dsdgp.layers])
#         KL_beta = self._build_beta_kl()
#         _, gamma_mean, gamma_var = self.dsdgp._build_predict(self.x_tr, full_cov=False, S=self.num_samples)
#         inner_exp = self.y_tr - tf.matmul(self.f_tr, self.m_beta) - gamma_mean
#         E_log_p_Y = -0.5 * self.num_data * tf.log(2 * np.pi * self.variance)
#         E_log_p_Y -= 0.5 / self.variance * tf.matmul(inner_exp, inner_exp, transpose_a=True)
#         lik = tf.reduce_mean(E_log_p_Y, axis=0)
#         lik -= 0.5 / self.variance * (tf.reduce_sum(tf.reduce_mean(gamma_var, axis=0)) + \
#                                       tf.trace(tf.matmul(tf.matmul(self.f_tr, tf.matrix_diag(self.s_beta)), \
#                                                          self.f_tr, transpose_b=True)))
#         lik -= KL_gamma + KL_beta
#         return tf.reduce_sum(lik)

    def _build_likelihood(self, X, Y):
        
        # PRIOR
        F = tf.cast(tf.concat((tf.ones((self.num_data, 1)), X), axis=1), dtype=tf.float32)
        KL_gamma = tf.reduce_sum([layer.KL() for layer in self.dsdgp.layers])
        KL_beta = self._build_beta_kl()
        
        # LOG-LIKELIHOOD
        _, gamma_mean, gamma_var = self.dsdgp._build_predict(X, full_cov=False, S=self.num_samples)
        inner_exp = Y - tf.matmul(F, self.m_beta) - gamma_mean
        E_log_p_Y = -0.5 * self.num_data * tf.log(2 * np.pi * self.variance)
        E_log_p_Y -= 0.5 / self.variance * tf.matmul(inner_exp, inner_exp, transpose_a=True)
        lik = tf.reduce_mean(E_log_p_Y, axis=0)
        
        # ELBO = LOG-LIKELIHOOD + PRIOR
        elbo = lik
        elbo -= 0.5 / self.variance * (tf.reduce_sum(tf.reduce_mean(gamma_var, axis=0)) + \
                                      tf.trace(tf.matmul(tf.matmul(F, tf.matrix_diag(self.s_beta)), \
                                                         F, transpose_b=True)))
        elbo -= KL_gamma + KL_beta
        return tf.reduce_sum(elbo), tf.reduce_sum(lik)

    def _build_beta_kl(self):
        kl = 0.5 * tf.reduce_sum(self.s_beta)
        kl += 0.5 * tf.matmul(self.m_beta, self.m_beta, transpose_a=True)
        kl -= 0.5 * self.input_dim
        kl -= 0.5 * tf.log(tf.reduce_prod(self.s_beta))
        return tf.reduce_sum(kl)

    def _build_predict(self, xnew, num_samples):
        xnew = xnew.astype(np.float32)
        tilde_xnew = np.concatenate((np.ones((xnew.shape[0], 1)), xnew), axis=1).astype(np.float32)
        fmean = self.dsdgp.predict_y(xnew, num_samples)

        S, D = tf.cast(fmean.shape[0], tf.int32), tf.cast(self.m_beta.shape[0], tf.int32)
        z = tf.random_normal([S, D, 1], dtype=tf.float32)
        beta = self.m_beta + z * tf.expand_dims(self.s_beta, -1) ** 0.5
        tilde_xnew = tf.tile(tf.expand_dims(tilde_xnew, 0), [S, 1, 1])
        noise = tf.sqrt(self.variance) * tf.random_normal(fmean.shape, dtype=tf.float32)
        
        return fmean + tf.matmul(tilde_xnew, beta) + noise