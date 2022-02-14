import numpy as np
from sklearn.utils import shuffle

def getData(objs, labels, data, k):
    objID_col = labels[:,3]
    tot_data = []
    for obj in objs:
        idx = np.where(objID_col == obj)
        tot_data.append( data[idx] )
    tot_data = np.concatenate(tot_data,axis=0)
    return tot_data, np.ones(tot_data.shape[0])*k

def getCL2Objs(labels):
    clID_col = labels[:,2]
    clID = np.unique(clID_col)
    hashClID2obj = {}
    for val in clID:
        idx = np.where(clID_col == val)
        t_labels = labels[idx]
        hashClID2obj[val] = np.unique( t_labels[:,3] )
    return hashClID2obj

data =np.load("data.npy")
labels = np.load("labels.npy")
hashClID2obj = getCL2Objs(labels)
train_perc = .5
train_valid = .2

tot_train_x = []
tot_train_y = []
tot_valid_x = []
tot_valid_y = []
tot_test_x = []
tot_test_y = []

for k in hashClID2obj.keys():
    objIds = hashClID2obj[k]
    objIds = shuffle(objIds)
    limit_train = int(len(objIds)* train_perc )
    limit_valid = limit_train + int(len(objIds)* train_valid)

    train_obj = objIds[0:limit_train]
    valid_obj = objIds[limit_train:limit_valid]
    test_obj = objIds[limit_valid::]

    train_x, train_y = getData(train_obj, labels, data, k)
    tot_train_x.append(train_x)
    tot_train_y.append(train_y)

    valid_x, valid_y = getData(valid_obj, labels, data, k)
    tot_valid_x.append(valid_x)
    tot_valid_y.append(valid_y)

    test_x, test_y = getData(test_obj, labels, data, k)
    tot_test_x.append(test_x)
    tot_test_y.append(test_y)

np.save("x_train.npy", np.concatenate(tot_train_x,axis=0))
np.save("y_train.npy", np.concatenate(tot_train_y,axis=0))
np.save("x_valid.npy", np.concatenate(tot_valid_x,axis=0))
np.save("y_valid.npy", np.concatenate(tot_valid_y,axis=0))
np.save("x_test.npy", np.concatenate(tot_test_x,axis=0))
np.save("y_test.npy", np.concatenate(tot_test_y,axis=0))
