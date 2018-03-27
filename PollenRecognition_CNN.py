import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import numpy as np
import theano
import os
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils, generic_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import KFold, StratifiedKFold 
from collections import Counter
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import keras.callbacks
from keras.callbacks import EarlyStopping
import gc
from sklearn.grid_search import ParameterGrid
import copy
from shutil import copyfile
import sys
# In[2]:

num_types=sys.argv[1]
input_path=sys.argv[2]
output_path=sys.argv[3]
losses_output_path=sys.argv[4]
filename = open(losses_output_path+"/conv_net_output", "w")
pollen = pickle.load( open( input_path+'/img4AugDS'+num_types, 'rb' ) )
target = pickle.load( open( input_path+'/trgt4AugDS'+num_types, 'rb' ) )
nb_classes=target.max()+1
output_path

# In[3]:

import warnings
warnings.filterwarnings('ignore')


# In[4]:

def relu(x):
    return theano.tensor.switch(x<0, 0, x)


# In[5]:

def tanh(x):
    return theano.tensor.tanh(x)


# In[6]:

counter = Counter(target)
counter
pollen.shape


# In[7]:

batch_size = int(sys.argv[5])
nb_epoch = int(sys.argv[6])

# input image dimensions
img_rows, img_cols = pollen.shape[1],pollen.shape[2]
print(img_rows,img_cols)
img_channels = 3
means_sc=[]
means_ac=[]
cv=0
#StratifiedKFold(y=target, n_folds=5, shuffle=True, random_state=23)
pollen=np.rollaxis(pollen,3,1)


# In[8]:

from keras import backend as K
K.set_image_dim_ordering('th')

def create_model(dropout_rate=0.5, optimizer=Adadelta, lr=1e-2, nb_filters=6, coef_filters=2.0,kernel=7, reduce_kernel=1, 
                nb_conv=3, nb_dense=2, dense_neurons=50, activation='sigmoid'):
        optimizer = optimizer(lr=lr)    
        print(nb_filters, kernel)
        model = Sequential()          
        model.add(Convolution2D(nb_filters, kernel, kernel,
                        border_mode='same',dim_ordering='th',
                        input_shape=(img_channels, img_rows, img_cols)))
       
        model.add(Activation(activation))
        nb_filters=int(nb_filters*coef_filters)
        if reduce_kernel and kernel>4:
            kernel=kernel-2        
        for i in range(nb_conv-1):
            model.add(Convolution2D(nb_filters, kernel, kernel))
            nb_filters=int(nb_filters*coef_filters)
            if reduce_kernel and kernel>4:
                kernel=kernel-2 
            model.add(Activation(activation))  
        model.add(Dropout(0.5))
        model.add(Flatten())    
        for i in range(nb_dense-1):
            model.add(Dense(dense_neurons))
            if dense_neurons>(nb_classes*2):
                dense_neurons=dense_neurons//2
            model.add(Dropout(dropout_rate))    
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        #model.optimizer.lr.set_value(lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])    
        return model

# In[10]:
search_space  = {"dropout_rate": [0.0, 0.2, 0.4, 0.6, 0.8],
              "optimizer": {Adam : {'lr' : [0.00001, 0.0001, 0.000005]},
                   Adadelta : {'lr' : [0.00001, 0.0001, 0.000005]},
                   Adagrad : {'lr' : [0.0001, 0.001, 0.00001]}
                   },
              "nb_filters": [6, 8, 10, 12, 14, 16],
              "coef_filters": [1.6, 2.0, 2.4, 2.8, 3.0, 3.4],
              "kernel": [3, 5, 7, 9],
              "reduce_kernel": [0, 1],
              "nb_conv": [3, 4, 5, 6, 7],
              "nb_dense": [2, 3, 4],
              "dense_neurons": [25, 50, 70, 100],
              "activation": ['relu',  'sigmoid', 'hard_sigmoid']
              }
param_grid = ParameterGrid(search_space)
all_params = []
for p in param_grid:
    all_params.append(p)
for key in search_space.keys():
    if (isinstance(search_space[key], dict)):
        new_params=[]
        for param in all_params:
            if (search_space[key][param[key]] is None):
                new_params.append(param)
            else:
                param_grid = ParameterGrid(search_space[key][param[key]])
                add_params = [p for p in param_grid]
                for aparam in add_params:
                    tparam = copy.copy(param)
                    tparam.update(aparam)
                    new_params.append(tparam)
        all_params = new_params      
grid_score=[]
grid_acc=[]
for param in all_params:  
    means_sc=[]
    means_ac=[]
    pred=1000
    checkpointer = ModelCheckpoint(filepath=output_path+"/weightsGridSearch"+num_types+".hdf5", verbose=1, save_best_only=True)
    kf = KFold(len(target), n_folds=5, shuffle=True, random_state=23)
    for train, test in kf:        
        model = create_model(**param)
        X_train = pollen[train]
        X_test = pollen[test]
        y_train = target[train]
        y_test = target[test]
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)    
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)   
        model.fit(X_train, Y_train, show_accuracy=True, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, 
                  validation_data=(X_test, Y_test), 
                  callbacks=[checkpointer, EarlyStopping(monitor='loss', min_delta=0, patience=20, mode='min')])
        score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        means_sc.append(score[0])
        means_ac.append(score[1])
    mean_sc=np.mean(means_sc)
    mean_ac=np.mean(means_ac)
    if mean_sc<pred:
        best_score=mean_sc
        best_acc=mean_ac
        best_params=param
        copyfile(output_path+"/weightsGridSearch"+num_types+".hdf5", output_path+"/weightsGridSearchBest"+num_types+".hdf5")
        pred=score[0]
    print("Test score %f, test acc %f with: %r" % (mean_sc, mean_ac, param))
    print("Test score %f, test acc %f with: %r" % (mean_sc, mean_ac, param), file=filename)    
    grid_score.append(mean_sc)
    grid_acc.append(mean_ac)    
    del(model)
    os.remove(output_path+"/weightsGridSearch"+num_types+".hdf5")
    gc.collect(2)   
    gc.collect(1)
    gc.collect(0)
    len(gc.get_objects())
'''for mean_sc, mean_ac, param in zip(grid_score, grid_acc, all_params):    
    print("Test score %f, test acc %f with: %r" % (mean_sc, mean_ac, param))
    print("Test score %f, test acc %f with: %r" % (mean_sc, mean_ac, param), file=filename)'''
print("Best: %f,%f using %r" % (best_score, best_acc, best_params))
print("Best: %f,%f using %r" % (best_score, best_acc, best_params), file=filename)
filename.close()

