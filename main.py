import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utils import construct_layer, calculate_metrics
from dset import step, sine, currin, park
from dsdgp import DSDGP
from lowdgp import LOWDGP
from imdgp import IMDGP
from tqdm.auto import tqdm
import os

def main(dstype, rep=10):
    """
    :param dstype: type of data-set i.e. "step", "sine", "currin", "park" (str)
    :param rep: number of replicate (int)
    :return: 3 numpy arrays for test metrics (mnll, r2, rmse) from each replicate (tuple)
    """
    REPLICATE = rep
    mnll, r2, rmse = [], [], []
    dset = step() if dstype=="step" else sine() if dstype=="sine" \
        else currin() if dstype=="currin" else park()
    
    ovars = [] # set of old variables which is used in previous replicates
    
    for itr in tqdm(range(REPLICATE), desc='Replicate'):
        
        print (itr, "-th experiment starts")
        
        # construct the data-set
        np.random.seed(itr)
        numlow, numhigh = dset.numlow, dset.numhigh
        pick_interval, start = dset.pick_interval, dset.start
#         numinducing1, numinducing2 = 10, 3
        numinducing1, numinducing2 = 6, 5

        test = np.random.uniform(dset.x_low, dset.x_high, [200, len(dset.x_low)]) # N of test observations = 200
        xl = np.random.uniform(dset.x_low, dset.x_high, [numlow, len(dset.x_low)]) # N of low-accuracy observations = 30
        xh = xl[start::pick_interval, :] # N of high-accuracy observation = 6

        yl, yh, = dset.low_fidelity(xl), dset.high_fidelity(xh)
        yl2, yh2 = dset.low_fidelity(test), dset.high_fidelity(test)

        # construct the low-accuracy model
        low_layers = construct_layer("rbf", [len(dset.x_low), 2, 1], numinducing1, None) # construct the layers for low-dgp
        lowdgp = LOWDGP(xl, yl, low_layers, 20)
        low_model_parameters = [var for var in tf.trainable_variables() if var not in ovars]
        ovars += low_model_parameters
        lik_low = lowdgp._build_likelihood()
        
        adjust_layers = construct_layer("rbf", [len(dset.x_low), 2, 2], numinducing2,
                                     None)
        imdgp = IMDGP(xh, yh, yl[start::pick_interval, :], adjust_layers, 20)
        adjust_model_parameters = [var for var in tf.trainable_variables() if var not in ovars]
        ovars += adjust_model_parameters
        lik_high = imdgp._build_likelihood()

        print ("Length of param :", len(low_model_parameters), len(adjust_model_parameters))
        print ("Training Details : lr_low = ", dset.learning_rate_low, ", lr_high = ", dset.learning_rate_high, ", iteration_low = " \
               , dset.iteration_low, ", iteration_high = ", dset.iteration_high)

        lik = lik_low + lik_high
        lr_low, lr_high = dset.learning_rate_low, dset.learning_rate_high
        training_low_param = tf.train.AdamOptimizer(lr_low).minimize(-lik_low, var_list=low_model_parameters)
        training_adjust_param = tf.train.AdamOptimizer(lr_high).minimize(-lik_high, var_list=adjust_model_parameters)
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        sess = tf.Session(config=config)
#         sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        np.random.seed(0)
        tf.set_random_seed(1234)
        # train the adjustment model
        for _ in tqdm(range(dset.iteration_low)):
            sess.run(training_low_param)
            
        pred_low = sess.run(lowdgp._build_predict(test, 50))
        pred_low_mean = np.expand_dims(pred_low.mean(axis=0), axis=0)
            
        for _ in tqdm(range(dset.iteration_high)):
            sess.run(training_adjust_param)
            
        # evaluate the model
        pred_high = sess.run(imdgp.predict_y(test, pred_low_mean, num_samples=50))
        y_mean_prediction = pred_high.mean(axis=0)
        y_var_prediction = pred_high.var(axis=0)
        metrics = calculate_metrics(yh2, y_mean_prediction, y_var_prediction)
        rmse.append(metrics['rmse'])
        r2.append(metrics['r2'])
        mnll.append(metrics['mnll'])

        print (metrics)
        sess.close()
    return np.array(rmse), np.array(r2), np.array(mnll)

if __name__ == "__main__":
        
    rmse, r2, mnll = main(dstype="currin", rep=10) # change these inputs if you want change data-set and N of replicates
    print ("RMSE (mean) : ", rmse.mean(), " / (std) : ", rmse.std())
    print ("R2 (mean) : ", r2.mean(), " / (std) : ", r2.std())
    print ("MNLL (mean) : ", mnll.mean(), " / (std) : ", mnll.std())
    
    import time
    name = str(time.time())
    file = open(name + ".csv", 'w')
    file.write("RMSE (mean) : " + str(rmse.mean()) + " / (std) : " + str(rmse.std()) + "\n")
    file.write("R2 (mean) : " + str(r2.mean()) + " / (std) : " + str(r2.std()) + "\n")
    file.write("MNLL: (mean) : " + str(mnll.mean()) + " / (std) : " + str(mnll.std()) + "\n")
    file.close()