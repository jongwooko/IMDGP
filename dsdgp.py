from layer import Layer_rbf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class DSDGP():
#     def __init__(self, X=None, Y=None, layers=None, Z=None, num_samples=1, name=""):
#         self.layers = layers
#         self.X, self.Y, self.num_samples = X, Y, num_samples
#         self.num_data = X.shape[0]
#         self.name=name
#         if Z is not None:
#             self.Z = Z
#         else:
#             self.Z = np.random.permutation(self.X.copy())[:layers[0].num_inducings]

    def __init__(self, num_data, input_dim, output_dim, layers=None, num_samples=1, name=""):
        self.num_data, self.num_samples = num_data, num_samples
        self.input_dim, self.output_dim = input_dim, output_dim
        self.name = name
        self.layers = layers
        self.Z = np.zeros((self.layers[0].num_inducings, self.input_dim))

    def initialize_param(self):
        X = np.random.random((self.num_data, self.input_dim))
        X_running, Z_running = self.layers[0].initialize_forward(X, self.Z)
        if len(self.layers) > 1:
            for layer in self.layers[1:]:
                X_running, Z_running = layer.initialize_forward(X_running, Z_running)

    def propagate(self, X, full_cov=False, S=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])

        Fs, Fmeans, Fvars = [], [], []

        F = sX
        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    def _build_predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fs[-1], Fmeans[-1], Fvars[-1]

    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)

    def predict_f_full_cov(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=True, S=num_samples)

    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)

    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples)

    def predict_y(self, Xnew, num_samples):
        F, Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        return tf.identity(F)