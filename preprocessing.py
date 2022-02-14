import rasterio as rio
import numpy as np

def local_normalize(bands):
    min = np.amin(bands)
    max = np.amax(bands)
    return (bands-float(min)) / (max-min)

def normalize(dataS2, nbands):
    _,_,totbands = dataS2.shape
    dataS2new = np.zeros(dataS2.shape)
    #print(dataS2new)
    #exit()
    for i in range(nbands):
        idxs_ch = np.arange(i,totbands,nbands)
        dataS2new[:,:,idxs_ch] = local_normalize(dataS2[:,:,idxs_ch])
    return dataS2new

def readMonoData(path):
    src = rio.open(path)
    band = src.read(1)
    src.closed
    return band

def readData(path):
    src = rio.open(path)
    bands = src.read()
    src.closed
    return np.moveaxis(bands, 0,-1)

#data_path = "/mnt/SSD/EMMANUEL/BDD/Koumbia/2018/S2_stack_GAPF_2018.tif"
year = "2021"
data_path = "/mnt/SSD/EMMANUEL/BDD/Koumbia/"+year+"/S2_stack_GAPF_"+year+".tif"

gt_path = "/home/emmanuel/These/DAAL4LMOD_01_03/"
clID_path = gt_path+year+"_Class-ID.tif"
objID_path = gt_path+year+"_Object-ID.tif"



#clID_path = gt_path+"2018_Class-ID.tif"
#objID_path = gt_path+"2018_Object-ID.tif"

clID = readMonoData(clID_path)
objID = readMonoData(objID_path)

n_bands = 4
dataS2 = readData(data_path)
dataS2 = normalize(dataS2, n_bands)
nrow, ncol, totTSBands = dataS2.shape

vals = np.unique(clID)
labels = []
tab_data = []
for id_ in vals[1::]:
    idx = np.where(clID == id_)
    rows, cols = idx
    for row_, col_ in zip(rows,cols):
        labels.append( [row_,col_,id_,objID[row_,col_]] )
        pixel = dataS2[row_,col_,:]
        temp_pixel = np.split(pixel, totTSBands/n_bands)
        tab_data.append( np.stack(temp_pixel,0) )

labels = np.array(labels)
tab_data = np.array(tab_data)

np.save("labels.npy",labels)
np.save("data.npy",tab_data)
