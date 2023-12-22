import torch
import sys
import os
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
from fno_utils.LOAD_fracture import load_fracture_mesh_strain_ds
import matplotlib.pyplot as plt
import numpy as np
import datetime



#import imgaug.augmenters as iaa

inference = True

# ===============
# check GPU, cuda
# ===============

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# ===============
# Hyperparameters
# ===============
# YOUR HYPERPARAMETERS

# list of directories for stress/damage fields
lst_dir0 = list()
lst_idx_UC = list()


if inference==True: #unseen data
    shuffle = False
    
    lst_dir0.append( r"D:\data\unseen\L0.05_vf0.6\h0.0002" )
    lst_dir0.append( r"D:\data\unseen\L0.05_vf0.25\h0.0002" )
    dir_mesh = r"D:\data\unseen\mesh"    
    
else:  #train data
    shuffle = True
    
    lst_dir0.append( r"D:\data\L0.05_vf0.3\h0.0002" )
    lst_dir0.append( r"D:\data\L0.05_vf0.5\h0.0002" )
    dir_mesh = r"D:\data\mesh"

## data augmentation
#transform = iaa.Sequential(
#    [
#        iaa.TranslateX(percent=(0.,0.99), mode="wrap"),
#        iaa.TranslateY(percent=(0.,0.99), mode="wrap"),
#    ]
#)
fieldtype = "stress"
downsample_ratio = "2"
# load the data
train_loader, test_loaders, output_encoder = load_fracture_mesh_strain_ds(
                         lst_dir0=lst_dir0,
                         dir_mesh=dir_mesh,
                         n_train=20, n_tests=[5],
                         batch_size=16, test_batch_sizes=[16],
                         test_resolutions=[251],
                         train_resolution=251,
                         grid_boundaries=[[0,1],[0,1]],
                         positional_encoding=True,
                         encode_input=False,
                         encode_output=False,
                         encoding='channel-wise')



# create a tensorised FNO model
model = TFNO( n_modes=(251, 251), 
              in_channels=3, out_channels=3,
              hidden_channels=32, projection_channels=64, 
              factorization='tucker', rank=0.42)

# load the model
model.load_state_dict(torch.load('C:/Users/Elliot/Documents/fyp/fno/stressmodel.pth'))
model = model.to(device)

n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

# create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-5, 
                                weight_decay=1e-6)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)

# creating the losses
l2loss = LpLoss(d=2, p=2, reduce_dims=[0,1])
h1loss = H1Loss(d=2, reduce_dims=[0,1])

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%
print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

'''
# Create the trainer
trainer = Trainer(model, n_epochs=7,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# Train the model
trainer.train(train_loader, test_loaders,
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

# save the model
#torch.save(model.state_dict(), '../model/fno/model.pth')
'''




# %%
# Plot the prediction, and compare with the ground-truth 

test_samples = test_loaders[251].dataset
model.cpu()


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'font.size': 8})

fig = plt.figure(figsize=(11, 11))
for index in range(3):
    data = test_samples[index+0]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))
    #
    ax1 = fig.add_subplot(3, 5, index*5 + 1)
    im = ax1.imshow(x[0], cmap='gray') ###
    if index == 0: 
        ax1.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])
    ###
    
    ###
    ax2 = fig.add_subplot(3, 5, index*5 + 2)
    im = ax2.imshow(y[0].squeeze(), cmap = 'rainbow', norm = mpl.colors.Normalize(vmin = -0.25, vmax = 1.95))
    if index == 0: 
        ax2.set_title('Ground-truth y1')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax = ax2, fraction = 0.050, pad = 0.04)
    #
    ax3 = fig.add_subplot(3, 5, index*5 + 3)
    im = ax3.imshow(out[0,0].squeeze().detach().numpy(), cmap = 'rainbow', norm = mpl.colors.Normalize(vmin = -0.25, vmax = 1.95))
    if index == 0: 
        ax3.set_title('Model Prediction')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax = ax3, fraction = 0.050, pad = 0.04)
    #
    ax4 = fig.add_subplot(3, 5, index*5 + 4)
    im = ax4.imshow(y[1].squeeze(), cmap = 'rainbow', norm = mpl.colors.Normalize(vmin = -0.25, vmax = 1.95))
    if index == 0: 
        ax4.set_title('Ground-truth y2')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax = ax4, fraction = 0.050, pad = 0.04)
    #
    ax5 = fig.add_subplot(3, 5, index*5 + 5)
    im = ax5.imshow(out[0,1].squeeze().detach().numpy(), cmap = 'rainbow', norm = mpl.colors.Normalize(vmin = -0.25, vmax = 1.95))
    if index == 0: 
        ax5.set_title('Model Prediction')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax = ax5, fraction = 0.050, pad = 0.04)
#cax = fig.add_axes([0.9, 0.13, 0.02, 0.70])
#fig.colorbar(im, orientation='vertical', cax=cax)

fig.suptitle('Inputs, Ground-truth Output and Model Prediction.', y=0.98)
plt.tight_layout()
# Save results
dt = datetime.datetime.now()
dt_string = dt.strftime("%d-%m-%Y_%H;%M;%S")
fig.savefig('C:/Users/Elliot/Pictures/fyp_images/FNOresults/' + ('{}_{}').format(fieldtype, dt_string))
#fig.show()


#fig.savefig('C:/Users/Elliot/Pictures/fyp_images/FNOresults/' + 'error' + ('_{}').format(dt_string))