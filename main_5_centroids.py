# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:34:20 2021

@author: emmanuel
"""

# importation des modules nécessaires
import sys

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from adda_nn_models import TempCNNClassifier
#from adda_nn_models import TempCNN_BN_Encoder, Classifier_BN, Discr
from sklearn.utils import shuffle
import time
from sklearn.cluster import KMeans
from scipy.stats import entropy

#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.40)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
def custom_l2_regularizer(weights):
    reg = 0
    for w in weights:
        reg+= tf.reduce_sum(tf.square(w))
    return reg

def ema(model, ema_model, ema_decay):
    for var, ema_var in zip(model.variables, ema_model.variables):
        if var.trainable:
            ema_var.assign((1 - ema_decay) * var + ema_decay * ema_var)
        else:
            ema_var.assign(tf.identity(var))


def getBatch(X, i, batch_size):
	start_id = (i*batch_size)
	t = (i+1) * batch_size
	end_id = min( (i+1) * batch_size, X.shape[0])
	batch_x = X[start_id:end_id]
	return batch_x


def trainingDANNModel(dann, optimizer, x_train, y_train, x_valid, y_valid, x_test, y_test, num_epochs, batch_size, loss_fn, model_file_name):
    epochs = range(num_epochs)
    iterations = x_train.shape[0] / batch_size
    if x_train.shape[0] % batch_size != 0:
        iterations += 1
    bestFM = 0

    for epoch in epochs:
        x_train, y_train = shuffle(x_train, y_train)
        tot_loss = 0.0
        start = time.time()
        for ibatch in range(int(iterations)):
            batch_x = getBatch(x_train, ibatch, batch_size)
            batch_y = getBatch(y_train, ibatch, batch_size)
            with tf.GradientTape() as tape:
                pred = dann(batch_x, training=True)
                loss_pred = loss_fn(batch_y, pred)
                tot_loss+=loss_pred
                grads = tape.gradient(tot_loss, dann.trainable_variables)
                optimizer.apply_gradients(zip(grads, dann.trainable_variables))

        end = time.time()

        #dann.save_weights("model_dann")
        pred_train = dann.predict(x_train)
        pred_valid = dann.predict(x_valid)
        pred_test = dann.predict(x_test)
        fscoreT = f1_score(y_train, np.argmax(pred_train,axis=1), average="weighted")
        fscoreV = f1_score(y_valid, np.argmax(pred_valid,axis=1), average="weighted")
        fscoreTe = f1_score(y_test, np.argmax(pred_test,axis=1), average="weighted")
        print("Epoch %d loss %.5f F-Score Train %.3f Valid %.3f Test %.3f with time %d" % (epoch, tot_loss/iterations, fscoreT, fscoreV, fscoreTe,(end-start)))
        if fscoreV > bestFM:
            bestFM = fscoreV
            print("-> -> BEST TEST score value :%f at epoch %d" % (fscoreTe, epoch) )

        sys.stdout.flush()

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")-1
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")-1
x_valid = np.load("x_valid.npy")
y_valid = np.load("y_valid.npy")-1


nb_class = len(np.unique(x_train))
#print(nb_class)
#exit()


# Instanciation d'une fonction objet à partir du modèle de l'entropie croisée pour des classes éparses et exclusives
loss_fn = keras.losses.SparseCategoricalCrossentropy()

#learning_rate= 0.0001
learning_rate= 0.0001
# Instanciation d'un optimiseur à partir du modèle de descente de gradient stochastique avec un taux d'apprentissage à 0.001
optimizer = keras.optimizers.Adam(learning_rate)


dann = TempCNNClassifier(nb_class)

model_file_name = "best_source_DANN"
batch_size = 32
num_epochs = 350
trainingDANNModel(dann, optimizer, x_train, y_train, x_valid, y_valid, x_test, y_test, num_epochs, batch_size, loss_fn, model_file_name)
