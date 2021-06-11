# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from dsdgp import DSDGP

class IMDGP():
#     def __init__(self, x_tr, yh_tr, yl_tr, layers=None, num_samples=20):
# #         self.rho_model = rho_model
# #         self.delta_model = delta_model

#         self.num_data = tf.cast(yh_tr.shape[0], dtype=tf.float32)
#         self.x_tr, self.num_samples = x_tr, num_samples
#         self.yh_tr, self.yl_tr = tf.cast(yh_tr, dtype=tf.float32), tf.cast(yl_tr, dtype=tf.float32)
#         self.layers = layers
#         self.variance = tf.exp(tf.Variable(tf.log(1e-4), dtype=tf.float32))
        
#         self.dsdgp = DSDGP(X=self.x_tr, Y=None, layers=self.layers, Z=None, num_samples=self.num_samples)
#         self.dsdgp.initialize_param()

    def __init__(self, num_data, input_dim, output_dim, layers, num_samples=20):
        self.num_data, self.num_samples = num_data, num_samples
        self.input_dim, self.output_dim = input_dim, output_dim
        self.layers = layers
        self.variance = tf.exp(tf.Variable(tf.log(1e-4), dtype=tf.float32))
        
        self.dsdgp = DSDGP(num_data=num_data, input_dim=input_dim, output_dim=output_dim, \
                           layers=layers, num_samples=num_samples)
        self.dsdgp.initialize_param()

    def _build_likelihood(self, X, YH, YL):
        
        # PRIOR
        KL = tf.reduce_sum([layer.KL() for layer in self.dsdgp.layers])
        _, mean, var = self.dsdgp._build_predict(X, full_cov=False, S=self.num_samples) #SND #SND
        
        # TODO : 이거 지금 mean으로 하는게 아닌 거 같다.
        rho_mean, delta_mean = tf.expand_dims(mean[:, :, 0], -1), tf.expand_dims(mean[:, :, 1], -1)        
        rho_var, delta_var = tf.expand_dims(var[:, :, 0], -1), tf.expand_dims(var[:, :, 1], -1)        
        
        # LOG-LIKELIHOOD
        inner_exp = YH - tf.multiply(rho_mean, YL) - delta_mean
        E_log_p_Y = -0.5 * self.num_data * tf.log(2 * np.pi * self.variance)
        E_log_p_Y -= 0.5 / self.variance * tf.matmul(inner_exp, inner_exp, transpose_a=True)
        lik = tf.reduce_mean(E_log_p_Y, axis=0)
        
        # ELBO = LOG-LIKELIHOOD + PRIOR
        elbo = lik
        elbo -= 0.5 / self.variance * (tf.reduce_sum(tf.reduce_mean(delta_var, axis=0)) + tf.reduce_sum(tf.multiply( \
            tf.square(YL), tf.reduce_mean(rho_var, axis=0)), axis=0))
        elbo -= KL
        return tf.reduce_sum(elbo), tf.reduce_mean(lik), tf.reduce_mean(elbo)

    def predict_y(self, Xnew, Ynew, num_samples):
        Ynew = Ynew.astype(np.float32)
        mean, _, _ = self.dsdgp._build_predict(Xnew, full_cov=False, S=num_samples) # SND #SND
        rho_mean, delta_mean = tf.expand_dims(mean[:, :, 0], -1), tf.expand_dims(mean[:, :, 1], -1)

        cat = np.random.choice(Ynew.shape[0], num_samples)
        yl_samples = []

        for choice in cat:
            yl_samples.append(Ynew[choice])
        yl_samples = np.stack(yl_samples, axis=0)
        yh_samples = tf.multiply(rho_mean, yl_samples) + delta_mean
        noise = tf.sqrt(self.variance) * tf.random_normal(yh_samples.shape, dtype=tf.float32)
        
        return yh_samples + noise