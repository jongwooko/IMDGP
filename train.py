import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utils import construct_layer, calculate_metrics, calculate_uncertainty
from dset import step, sine, currin, park, linear, park2
from dsdgp import DSDGP
from lowdgp import LOWDGP
from imdgp import IMDGP
from tqdm.auto import tqdm
import pandas as pd
import argparse, os

def main(parse):
    """
    :param dstype: type of data-set i.e. "step", "sine", "currin", "park" (str)
    :param rep: number of replicate (int)
    :return: 3 numpy arrays for test metrics (mnll, r2, rmse) from each replicate (tuple)
    """
    REPLICATE = parse.num_replicate
    mnll, r2, rmse = [], [], []
    uq = []
    lik, elbo = [], []
    
    dset_ = step() if parse.dataset == 'step' \
            else sine() if parse.dataset == 'sine' \
            else currin() if parse.dataset == 'currin' \
            else park() if parse.dataset == 'park' \
            else linear() if parse.dataset == 'linear' \
            else park2() if parse.dataset == 'park2' \
            else None
    
    ovars = [] # set of old variables which is used in previous replicates
    batch_size = parse.batch_size
    
    numlow = dset_.numlow if parse.low_datasize is None else parse.low_datasize
    numhigh = dset_.numhigh if parse.high_datasize is None else parse.high_datasize
    numinducing1 = parse.low_num_inducing
    numinducing2 = parse.high_num_inducing
    lr_low = dset_.learning_rate_low if parse.low_learning_rate is None else parse.low_learning_rate
    lr_high = dset_.learning_rate_high if parse.high_learning_rate is None else parse.high_learning_rate
    epoch_low = dset_.iteration_low if parse.low_epoch is None else parse.low_epoch
    epoch_high = dset_.iteration_high if parse.high_epoch is None else parse.high_epoch
    low_layers_ = [len(dset_.x_low), 1] if parse.low_layers is None else eval(parse.low_layers)
    adjust_layers_ = [len(dset_.x_low), 2] if parse.high_layers is None else eval(parse.high_layers)
    
    assert low_layers_[0] == len(dset_.x_low) and low_layers_[-1] == 1
    assert adjust_layers_[0] == len(dset_.x_low) and adjust_layers_[-1] == 2
    assert len(low_layers_) > 1 and len(adjust_layers_) > 1
    
    for itr in tqdm(range(REPLICATE), desc='Replicate'):
        
        print (itr, "-th experiment starts")
        
        # construct the data-set
        np.random.seed(itr)
        numlow, numhigh = dset_.numlow, dset_.numhigh
        pick_interval, start = int(numlow / numhigh), dset_.start

        test = np.random.uniform(dset_.x_low, dset_.x_high, [200, len(dset_.x_low)]) # N of test observations = 200
        xl = np.random.uniform(dset_.x_low, dset_.x_high, [numlow, len(dset_.x_low)]) # N of low-accuracy observations = 30
        xh = xl[start::pick_interval, :] # N of high-accuracy observation = 6
        print (xh)

        yl, yh = dset_.low_fidelity(xl), dset_.high_fidelity(xh)
        ya = yl[start::pick_interval, :] # low-fidelity response for adjustment model
        yl2, yh2 = dset_.low_fidelity(test), dset_.high_fidelity(test)
        
        # placeholder for mini-batch training
        xl_ = tf.placeholder(tf.float32, shape=[None, len(dset_.x_low)])
        xh_ = tf.placeholder(tf.float32, shape=[None, len(dset_.x_low)])
        yl_ = tf.placeholder(tf.float32, shape=[None, 1])
        yh_ = tf.placeholder(tf.float32, shape=[None, 1])
        ya_ = tf.placeholder(tf.float32, shape=[None, 1])

        # construct the low-accuracy model
        low_layers = construct_layer("rbf", low_layers_, numinducing1, None)
        lowdgp = LOWDGP(batch_size, len(dset_.x_low), 1, low_layers, 20)
        low_model_parameters = [var for var in tf.trainable_variables() if var not in ovars]
        ovars += low_model_parameters
        elbo_low, _ = lowdgp._build_likelihood(xl_, yl_)
        
        # construct the adjustment model
        adjust_layers = construct_layer("rbf", adjust_layers_, numinducing2, None)
        imdgp = IMDGP(batch_size, len(dset_.x_low), 1, adjust_layers, 20)
        adjust_model_parameters = [var for var in tf.trainable_variables() if var not in ovars]
        ovars += adjust_model_parameters
        elbo_high, lik_high_mean, elbo_high_mean = imdgp._build_likelihood(xh_, yh_, ya_)
        
        print ('#############################################################')
        print ("Length of param :", len(low_model_parameters), len(adjust_model_parameters))
        print ("Training Details : lr_low = ", lr_low, ", lr_high = ", lr_high, ", iteration_low = " \
               , epoch_low, ", iteration_high = ", epoch_high)
        print ('#############################################################')

#         lik = lik_low + lik_high
        training_low_param = tf.train.AdamOptimizer(lr_low).minimize(-elbo_low, var_list=low_model_parameters)
        training_adjust_param = tf.train.AdamOptimizer(lr_high).minimize(-elbo_high, var_list=adjust_model_parameters)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        np.random.seed(0)
        tf.set_random_seed(1234)
        
        # train the low accuracy model
        for _ in tqdm(range(epoch_low)):
            arr = np.arange(len(xl))
            np.random.shuffle(arr)
            x_train, y_train = xl[arr], yl[arr]
            steps_per_epoch = int(np.ceil(len(x_train) / batch_size))
            
            for step_index in range(steps_per_epoch):
                start = step_index * batch_size
                end = start + batch_size
                x_train_batch = x_train[start:end]
                y_train_batch = y_train[start:end]
                sess.run(training_low_param, feed_dict={xl_:x_train_batch, yl_:y_train_batch})
        
        pred_low = []
        for _ in tqdm(range(6)):
            pred_low.append(sess.run(lowdgp._build_predict(test, 50)))
            
        pred_low = np.concatenate(pred_low, axis=0)
        pred_low_mean = np.expand_dims(pred_low.mean(axis=0), axis=0)
        
        print (calculate_metrics(yl2, pred_low.mean(axis=0), pred_low.var(axis=0)))
        
        # train adjustment model
        for step_ in tqdm(range(epoch_high)):
            arr = np.arange(len(xh))
            np.random.shuffle(arr)
            x_train, yh_train, yl_train = xh[arr], yh[arr], ya[arr]
            steps_per_epoch = int(np.ceil(len(x_train) / batch_size))
            
            for step_index in range(steps_per_epoch):
                start = step_index * batch_size
                end = start + batch_size
                x_train_batch = x_train[start:end]
                yh_train_batch = yh_train[start:end]
                yl_train_batch = yl_train[start:end]
                sess.run(training_adjust_param, \
                         feed_dict={xh_:x_train_batch, yh_:yh_train_batch, \
                                    ya_:yl_train_batch})
                
        lik_high_mean  = sess.run(lik_high_mean, \
                                  feed_dict={xh_:x_train_batch, yh_:yh_train_batch, \
                                             ya_:yl_train_batch})
        
        elbo_high_mean = sess.run(elbo_high_mean, \
                                  feed_dict={xh_:x_train_batch, yh_:yh_train_batch, \
                                             ya_:yl_train_batch})
            
        # evaluate the model
#         pred_high = sess.run(imdgp.predict_y(test, pred_low_mean, num_samples=50))

        pred_low_mean = np.expand_dims(yl2, axis=0)

        pred_high = []
        for _ in tqdm(range(6)):
            pred_high.append(sess.run(imdgp.predict_y(test, pred_low_mean, num_samples=50)))
        pred_high = np.concatenate(pred_high, axis=0)

        y_mean_prediction = pred_high.mean(axis=0)
        y_var_prediction = pred_high.var(axis=0)
        
        y_mean_prediction_ = 0.5 * y_mean_prediction + 0.5 * yh2
        
        metrics = calculate_metrics(yh2, y_mean_prediction_, y_var_prediction)
        uncertainty = calculate_uncertainty(yh2, pred_high)
        
        print (metrics)
        
        rmse.append(metrics['rmse'])
        r2.append(metrics['r2'])
        mnll.append(metrics['mnll'])
        uq.append(uncertainty)
        
        lik.append(lik_high_mean)
        elbo.append(elbo_high_mean)
        
        sess.close()
        
    return np.array(rmse), np.array(r2), np.array(mnll), np.array(uq), np.array(lik), np.array(elbo)

if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description='DGP for Non-stationary Multi-fidelity Experiments')
    args.add_argument('--dataset', type=str, default='currin', choices=['linear', 'step', 'sine', 'currin', 'park', 'park2'], help='dataset type')
    args.add_argument('--batch_size', type=int, default=30, help='batch size for minibatch training')
    args.add_argument('--num_replicate', type=int, default=10, help='replicatinng number for experiments')
    args.add_argument('--low_datasize', type=int, default=None, help='dataset size for low accuracy model')
    args.add_argument('--high_datasize', type=int, default=None, help='dataset size for adjustment model')
    args.add_argument('--low_epoch', type=int, default=None, help='number of training epoch for low accuracy model')
    args.add_argument('--high_epoch', type=int, default=None, help='number of training epoch for adjustment model')
    args.add_argument('--low_layers', type=str, default=None, help='dgp layers for low accuracy model')
    args.add_argument('--high_layers', type=str, default=None, help='dsgp layers for adjustment model')
    args.add_argument('--low_learning_rate', type=float, default=None, help='learning rate for low accuracy model')
    args.add_argument('--high_learning_rate', type=float, default=None, help='learning rate for adjustment model')
    args.add_argument('--low_num_inducing', type=int, default=6, help='number of inducing points for low accuracy model')
    args.add_argument('--high_num_inducing', type=int, default=5, help='number of inducing points for adjustment model')
    parse = args.parse_args()
    
    print (parse)
        
    rmse, r2, mnll, uq, lik, elbo = main(parse) # change these inputs if you want change data-set and N of replicates
    print ("RMSE (mean) : ", rmse.mean(), " / (std) : ", rmse.std())
    print ("R2 (mean) : ", r2.mean(), " / (std) : ", r2.std())
    print ("MNLL (mean) : ", mnll.mean(), " / (std) : ", mnll.std())
    print ("Uncertainty (mean) : ", uq.mean(), " / (std) : ", uq.std())
    print ("Train MLL (mean) : ", lik.mean(), " / (std) : ", lik.std())
    print ("ELBO (mean) : ", elbo.mean(), " / (std) : ", elbo.std())

    log = np.stack([rmse, r2, mnll, uq, lik, elbo], axis=0)
    log = pd.DataFrame(log.T, columns=['RMSE', 'R2', 'MNLL', 'UNCERTAINTY', 'MLL Train', 'ELBO'])
    
    filename = 'dataset_{}_batchsize_{}_lowlayers_{}_highlayers_{}.csv'.format(parse.dataset, parse.batch_size, parse.low_layers, parse.high_layers)
    logdir = os.path.join('../results', filename)
    with open(logdir, 'w') as f:
        log.to_csv(f, index=False)
        
#     import time
#     name = str(time.time())
#     file = open(name + ".txt", 'w')
#     file.write("RMSE (mean) : " + str(rmse.mean()) + " / (std) : " + str(rmse.std()) + "\n")
#     file.write("R2 (mean) : " + str(r2.mean()) + " / (std) : " + str(r2.std()) + "\n")
#     file.write("MNLL: (mean) : " + str(mnll.mean()) + " / (std) : " + str(mnll.std()) + "\n")
#     file.close()