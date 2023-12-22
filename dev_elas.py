
import torch

from fno_utils.NN_warehouse import *
from fno_utils.train_structure_warehouse import *
from fno_utils.IOfcts import *
from fno_utils.helperFcts import *

import matplotlib.pyplot as plt
import numpy as np

# import cv2
import imgaug.augmenters as iaa


inference = True


# ===============
# check GPU, cuda
# ===============

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# ==============
# Hyperparamters
# ==============
portion_trains = 0.9
num_epochs = 2000
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5


# directory for initial geometry / mesh
dir_mesh = r"D:\data\mesh"

# list of directories for stress/damage fields
lst_dir0 = list()
lst_idx_UC = list()


shuffle = True
lst_dir0.append( r"D:\data\L0.05_vf0.3\h0.0002" )
lst_dir0.append( r"D:\data\L0.05_vf0.5\h0.0002" )
   
dir_mesh = r"D:\data\mesh"



# data augmentation
transform = iaa.Sequential(
    [
        iaa.TranslateX(percent=(0.,0.99), mode="wrap"),
        iaa.TranslateY(percent=(0.,0.99), mode="wrap"),
    ]
)
#if inference:
#    transform = None

# prepare the dataset (change to damage)
dataset = stepDataset(dir_mesh, lst_dir0, LOADprefix='Load0.0', transform=transform, outputFields='stress', varLOAD=False, step=10)

numDAT = len(dataset)
if inference == True:
    numDAT = 6960

idx = 123
mesh, load, field = dataset[idx]
fig,ax = plt.subplots(1,2)
ax[0].imshow(mesh[0]); ax[0].axis('off')
ax[1].imshow(field[0], cmap='jet'); ax[1].axis('off')
plt.show()


# ==================
# split train / test
# ==================

num_trains = int(len(dataset) * portion_trains)
num_tests = len(dataset) - num_trains
train_set, test_set = torch.utils.data.random_split(dataset, [num_trains, num_tests])

loaders = {
    'train': DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=shuffle),
    'test': DataLoader(dataset=test_set, num_workers=4, batch_size=batch_size, shuffle=shuffle),
}

mesh, load, field = train_set[11]


# =======================
# set training parameters
# =======================

init_epoch = 0

checkpoint_path = "../models/mesh2stress/current_bsiz"+str(batch_size)+"_lr"+str(learning_rate)+"_wdecay"+str(weight_decay)+"_dataAug_numDAT"+str(numDAT)+"_SA_2.pt"
best_model_path = "../models/mesh2stress/best_bsiz"+str(batch_size)+"_lr"+str(learning_rate)+"_dataAug_numDAT"+str(numDAT)+"_SA_2.pt"
dir_tb = '../runs/mesh2stress/bsiz'+str(batch_size)+'lr'+str(learning_rate)+'wdecay'+str(weight_decay)+'_dataAug_numDAT'+str(numDAT)+'_SA_2'

net = NN_mesh2stress3_SA_2()
#net = NN_mesh2stress3_SA_2_skip()

if use_cuda:
    net = net.cuda()

reconLoss = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ===============
# train the model
# ===============

if init_epoch > 0:
    # load the saved checkpoint
    net, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, net, optimizer)

if inference == False:
    #trained_model = train_mesh2stress(net, init_epoch, num_epochs, np.Inf, loaders,
    #                                     optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb, num_field=3)
    trained_model = train_mesh2stress_2(net, init_epoch, num_epochs, np.Inf, loaders,
                                         optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb, num_field=3)



# =============
# visualisation
# =============

# create a model
model = NN_mesh2stress3_SA_2() #using RVE as input
#model = NN_mesh2stress3_SA() #using Lame coefficients as input
model = model.cuda();   use_cuda = True
#model = model.cpu();    use_cuda = False

# load the saved checkpoint
model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer)

# ===================================
from fno_utils.inference import *
import time
import matplotlib.gridspec as gridspec

nsamples = 240

t_total = 0
F1_total = list()
for isample in range(nsamples):
    
    t0 = time.time()
    #isample = 123
    mesh, load, field = dataset[isample]
    mesh = torch.unsqueeze(mesh, 0)
    load = torch.unsqueeze(load, 0)
    field = torch.unsqueeze(field, 0)
    pred, loss = inference_mesh2stress_2(model, reconLoss, mesh, load, field, use_cuda)
    #pred, loss = inference_mesh2stress_inMESH(model, reconLoss, mesh, load, field)
    t0 = time.time() - t0
    t_total += t0
    
    '''
    # visu
    ibatch = 0
    v00, v01 = field[ibatch,0].min().numpy(), field[ibatch,0].max().numpy()
    v10, v11 = field[ibatch,1].min().numpy(), field[ibatch,1].max().numpy()
    v20, v21 = field[ibatch,2].min().numpy(), field[ibatch,2].max().numpy()
    v0 = [v00, v10, v20]
    v1 = [v01, v11, v21]
    
    plt.figure(figsize = (2, 4))
    gs1 = gridspec.GridSpec(2, 4)
    gs1.update(wspace=0.01, hspace=0.01, left=0.01, bottom=0.01, top=1, right=0.99)
    for i in range(8):
        ax1 = plt.subplot(gs1[i])
        plt.axis('off')
        if i==0:
            im=ax1.imshow(mesh[ibatch,0].cpu().T)
        elif i<4:
            im=ax1.imshow(field[ibatch,i-1].cpu().T, cmap='jet', vmin=v0[i-1], vmax=v1[i-1])
        elif i==4:
            continue
        else:
            im=ax1.imshow(pred[ibatch,i-5].cpu().T, cmap='jet', vmin=v0[i-5], vmax=v1[i-5])
        #plt.colorbar(im,ax=ax1,location='bottom', fraction=0.047)
    
    plt.show()
    '''
    
    # F1 score of segmented Invariant map
    pred = pred.cpu()
    I1 = (field[0,0]+field[0,1]+field[0,2])
    #I2 = 0.5* (I1**2-(field[0,0]**2+field[0,1]**2 + 2.*field[0,2]**2))
    #I3 = field[0,0]*field[0,1] - field[0,2]**2
    I1_p = (pred[0,0]+pred[0,1]+pred[0,2])
    #I2_p = 0.5* (I1_p**2-(pred[0,0]**2+pred[0,1]**2 + 2.*pred[0,2]**2))
    #I3_p = pred[0,0]*pred[0,1] - pred[0,2]**2
    
    ##histograms
    #plt.figure()
    #plt.hist(I1_p[mesh[0,0]!=3], **kwargs, color='red')
    #plt.show()
    
    #segmented
    true_hot = I1 > np.percentile(I1[mesh[0,0]!=3], 99)
    pred_hot = I1_p > np.percentile(I1_p[mesh[0,0]!=3], 99)
    
    TP = np.count_nonzero(pred_hot & true_hot)
    FP = np.count_nonzero(pred_hot & ~true_hot)
    FN = np.count_nonzero(~pred_hot & true_hot)
    F1 = 2*TP / (2*TP+FP+FN)
    
    ##visu
    #np.where(np.abs(F1_total-0.9)<0.05)
    #fig,ax=plt.subplots(1,2)
    #ax[0].imshow(~true_hot,cmap='gray'); ax[0].tick_params(labelbottom=False,labelleft=False)
    #ax[1].imshow(~pred_hot,cmap='gray'); ax[1].tick_params(labelbottom=False,labelleft=False)
    #plt.show()
     
    F1_total.append(F1)

print('average inference time: %f s\n'%(t_total/nsamples))
F1_total = np.array(F1_total)

# histograms of the F1 scores
kwargs = dict(bins=100, density=True, facecolor='g', alpha=0.7)
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 24}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(12,7))
n, bins, patches = ax.hist(F1_total, **kwargs)
ax.set_xlabel('F1 score')
ax.set_ylabel('Probability density')
ax.set_xlim(0, 1)
ax.grid(True)
plt.show()

#
np.where(np.abs(F1_total-0.4)<0.05)
#

####
#repeatability of the CNN prediction
####
isample = 212
ibatch = 0
F1_score = list()
for i in range(10):
    mesh, load, field = dataset[isample]
    mesh = torch.unsqueeze(mesh, 0)
    load = torch.unsqueeze(load, 0)
    field = torch.unsqueeze(field, 0)
    t0 = time.time()
    pred, loss = inference_mesh2stress_2(model, reconLoss, mesh, load, field)
    
    # F1 score of segmented Invariant map
    pred = pred.cpu()
    I1 = (field[0,0]+field[0,1]+field[0,2])
    I1_p = (pred[0,0]+pred[0,1]+pred[0,2])
    
    #segmented
    true_hot = I1 > np.percentile(I1[mesh[0,0]!=3], 99)
    pred_hot = I1_p > np.percentile(I1_p[mesh[0,0]!=3], 99)
    
    TP = np.count_nonzero(pred_hot & true_hot)
    FP = np.count_nonzero(pred_hot & ~true_hot)
    FN = np.count_nonzero(~pred_hot & true_hot)
    F1 = 2*TP / (2*TP+FP+FN)
    
    F1_score.append(F1)
    
    
F1_score = np.array(F1_score)

kwargs = dict(bins=100, density=True, facecolor='g', alpha=0.7)
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 24}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(12,6))
n, bins, patches = ax.hist(F1_score, **kwargs)
ax.set_xlabel('F1 score')
ax.set_ylabel('Probability density')
ax.set_xlim(0, 1)
ax.grid(True)
plt.show()


#
F1_score_58 = np.copy(F1_score)

fig, ax = plt.subplots(figsize=(12,6))
kwargs = dict(bins=100, density=True, alpha=0.7)
n, bins, patches = ax.hist(F1_score_4, **kwargs, color='red')
n, bins, patches = ax.hist(F1_score_58, **kwargs, color='green')
n, bins, patches = ax.hist(F1_score_212, **kwargs, color='blue')
n, bins, patches = ax.hist(F1_score_201, **kwargs, color='orange')
ax.set_xlabel('F1 score')
ax.set_ylabel('Probability density')
ax.set_xlim(0, 1)
ax.grid(True)
plt.show()


##
plt.figure(); plt.imshow(~true_hot.T,cmap='gray'); plt.show()
plt.figure(); plt.imshow(~pred_hot.T,cmap='gray'); plt.show()


