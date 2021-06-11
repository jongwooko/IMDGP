# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import scipy.stats

def reparameterize(mean, var, z, full_cov=False):
    if var is None:
        return mean
    if full_cov is False:
        return mean + z * (var + 1e-4) ** 0.5
    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2]
        mean = tf.transpose(mean, (0, 2, 1))
        var = tf.transpose(var, (0, 3, 1, 2))
        I = 1e-4 * tf.eye(N, dtype=tf.float32)[None,None,:,:]
        chol = tf.cholesky(var + I)
        z_SDN1 = tf.transpose(z, [0, 2, 1])[:,:,:,None]
        f = mean + tf.matmul(chol, z_SDN1)[:,:,:,0]
        return tf.transpose(f, (0, 2, 1))

def fit_shape(obsX1, obsX2):
    if len(obsX1.shape) == 2 and len(obsX2.shape) == 2:
#         numDataPoints1 = len(obsX1) if type(obsX1) == list else obsX1.shape[0]
#         numDataPoints2 = len(obsX2) if type(obsX2) == list else obsX2.shape[0]
        numDimension = len(obsX1[0]) if type(obsX1) == list else obsX1.shape[1]

        obsX1 = tf.reshape(obsX1, shape=[-1, 1, numDimension])
        obsX2 = tf.reshape(obsX2, shape=[-1, numDimension])

    elif len(obsX1.shape) == 2 and len(obsX2.shape) == 3:
#         numDataPoints1 = len(obsX1) if type(obsX1) == list else obsX1.shape[0]
#         numDataPoints2 = len(obsX2[0]) if type(obsX2) == list else obsX2.shape[1]
        numSamplePoints2 = len(obsX2) if type(obsX2) == list else obsX2.shape[0]
        numDimension = len(obsX1[0]) if type(obsX1) == list else obsX1.shape[1]

        obsX1 = tf.reshape(obsX1, shape=[1, -1, 1, numDimension])
        obsX2 = tf.reshape(obsX2, shape=[numSamplePoints2, 1, -1, numDimension])

    elif len(obsX1.shape) == 3 and len(obsX2.shape) == 3:
#         numDataPoints1 = len(obsX1[0]) if type(obsX1) == list else obsX1.shape[1]
#         numDataPoints2 = len(obsX2[0]) if type(obsX2) == list else obsX2.shape[1]
        numSamplePoints1 = len(obsX1) if type(obsX1) == list else obsX1.shape[0]
        numSamplePoints2 = len(obsX2) if type(obsX2) == list else obsX2.shape[0]
        numDimension = len(obsX1[0][0]) if type(obsX1) == list else obsX1.shape[2]

        obsX1 = tf.reshape(obsX1, shape=[numSamplePoints1, -1, 1, numDimension])
        obsX2 = tf.reshape(obsX2, shape=[numSamplePoints2, -1, numDimension])

    return obsX1, obsX2, numDimension

def kernel_rbf(theta, obsX1, obsX2):
    obsX1, obsX2, numDimension = fit_shape(obsX1, obsX2)
    alpha = tf.reshape(theta[1::], shape=[1, numDimension])
    k = tf.squared_difference(tf.to_float(obsX1), tf.to_float(obsX2))
    kk = tf.reduce_sum((tf.to_float(alpha) * k), axis=-1)
    matCovariance = theta[0] * tf.exp((-0.5) * kk)
    return matCovariance

def kernel_linear(sd_diag_matrix, obsX1, obsX2):
    A = tf.matrix_diag(sd_diag_matrix)
    A, obsX1, obsX2 = tf.cast(A, tf.float32), tf.cast(obsX1, tf.float32), tf.cast(obsX2, tf.float32)
    matCovariance = tf.matmul(tf.matmul(obsX1, A), obsX2, transpose_b=True)
    return tf.cast(matCovariance, tf.float32)

def construct_layer(type, features, numinducing, mean_function=None):
    """
    :param type: string, kernel type i.e., "rbf" (other can be updated)
    :param features: list, feature of each layers.
    ex1) [4, 2, 1] - input : 4, hidden : 2, output : 1
    ex2) [4, 2, 2, 1] - input : 4, hidden1 : 2, hideen2 : 2, output : 1
    :param numinducing: int, number of inducing points
    :return: list, layers
    """
    from layer import Layer_rbf
    layers = []
    for layer in range(len(features)-1):
        if type == "imdgp":
            layers.append(Layer_imdgp(features[layer], features[layer+1], numinducing, mean_function))
        elif type == "rbf":
            layers.append(Layer_rbf(features[layer], features[layer+1], numinducing, mean_function))
    return layers

def calculate_metrics(y_test, y_mean_prediction, y_var_prediction):
    # R2
    r2 = r2_score(y_test, y_mean_prediction)
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_mean_prediction))
    # Test log likelihood
    mnll = -np.sum(scipy.stats.norm.logpdf(y_test, loc=y_mean_prediction, scale=np.sqrt(y_var_prediction)))/len(y_test)
    return {'r2': r2, 'rmse': rmse, 'mnll': mnll}

def calculate_uncertainty(y_test, y_pred):
    
    upper = np.quantile(y_pred, 0.975, axis=0)
    lower = np.quantile(y_pred, 0.025, axis=0)
    
    results = (y_test <= upper) & (y_test >= lower)
    return results.mean()
    