import matplotlib.pyplot as plt
import numpy as np

# ================
# helper functions
# ================
def imshow_multiple(dataset, nimgs, inout='input'):
    if inout == 'input':
        idx = 0
        ncols = 1
    elif inout == 'output':
        idx = 1
        ncols = 1
    elif inout == 'inout':
        idx = 0
        ncols = 2

    for i in range(nimgs):
        sample = dataset[i]
        ax = plt.subplot(ncols, nimgs, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample[idx].transpose(0, -1))

        if inout == 'inout':
            ax = plt.subplot(ncols, nimgs, nimgs+i + 1)
            ax.set_title('Sample #{}'.format(i))
            plt.imshow(sample[1].transpose(0, -1))

    plt.show()


def imshow_multichannel(dataset, idx):

    sample = dataset[idx]
    nchannels = np.shape(sample[1])[0]

    fig, ax = plt.subplots(nrows=1, ncols=nchannels+1)

    # plot mesh
    # ax = plt.subplot(1, nchannels+1, 1)
    ax[0].imshow(sample[0].transpose(0, -1))

    # plot multichannel output
    for j in range(nchannels):
        ax[j+1].imshow(sample[1][j,:,:])

    fig.suptitle('Sample #%d' % (idx))



def imshow_elastic2crackDATA(dataset, idx):

    sample = dataset[idx]
    nc_in = np.shape(sample[0])[0]
    nc_out = np.shape(sample[1])[0]

    fig, ax = plt.subplots(nrows=1, ncols=nc_in+nc_out)

    # plot mesh
    # ax = plt.subplot(1, nchannels+1, 1)
    for j in range(nc_in):
        ax[j].imshow(sample[0][j,:,:],
                     vmin=sample[0][j,:,:].flatten().min(), vmax=sample[0][j,:,:].flatten().max())

    # plot multichannel output
    for j in range(nc_out):
        ax[j+nc_in].imshow(sample[1][j,:,:])

    fig.suptitle('Sample #%d' % (idx))


def imshow_incremental(input, output, data_macro=None):

    mesh = input[0]
    load = input[1]
    stress_in = input[2]
    damage_in = input[3]
    stress_out = output[0]
    damage_out = output[1]

    #move axis if torch tensor
    if mesh.shape[0]==1:
        mesh = np.moveaxis(mesh.numpy(),0,-1)
        stress_in = np.moveaxis(stress_in.numpy(), 0, -1)
        damage_in = np.moveaxis(damage_in.numpy(), 0, -1)
        stress_out = np.moveaxis(stress_out.numpy(), 0, -1)
        damage_out = np.moveaxis(damage_out.numpy(), 0, -1)

    fig, ax = plt.subplots(nrows=2, ncols=3)
    im=ax[0,0].imshow(mesh); fig.colorbar(im, ax=ax[0,0]); ax[0,0].set_title('geometry')
    ax[0,0].axis('off')
    im=ax[0,1].imshow(stress_in[:,:,0],vmin=0,vmax=150); fig.colorbar(im,ax=ax[0,1]); ax[0,1].set_title('stress, load'+str(load[0]))
    ax[0,1].axis('off')
    im=ax[0,2].imshow(damage_in,vmin=0,vmax=1); fig.colorbar(im,ax=ax[0,2]); ax[0,2].set_title('damage, load'+str(load[0]))
    ax[0,2].axis('off')
    im=ax[1,1].imshow(stress_out[:,:,0],vmin=0,vmax=150); fig.colorbar(im,ax=ax[1,1]); ax[1,1].set_title('stress, load'+str(load[1]))
    ax[1,1].axis('off')
    im=ax[1,2].imshow(damage_out,vmin=0,vmax=1); fig.colorbar(im,ax=ax[1,2]); ax[1,2].set_title('damage, load'+str(load[1]))
    ax[1,2].axis('off')

    if data_macro is None:
        ax[1,0].axis('off')
    else:
        ax[1,0].plot(data_macro[:,7],data_macro[:,1])
        ax[1,0].plot(load[0],data_macro[data_macro[:,7]==load[0],1],'o')
        ax[1,0].plot(load[1],data_macro[data_macro[:,7]==load[1],1],'o')


def imshow_history(input, output, curve_macro=None):
    mesh = input[0]
    load = input[1]
    stress_in = input[2]
    damage_in = input[3]
    stress_out = output[0][:]
    damage_out = output[1][:]
    nsteps = len(stress_out)

    #move axis if torch tensor
    if mesh.shape[0]==1:
        mesh = np.moveaxis(mesh.numpy(),0,-1)
        stress_in = np.moveaxis(stress_in.numpy(), 0, -1)
        damage_in = np.moveaxis(damage_in.numpy(), 0, -1)
        for i in range(nsteps):
            stress_out[i] = np.moveaxis(stress_out[i].numpy(), 0, -1)
            damage_out[i] = np.moveaxis(damage_out[i].numpy(), 0, -1)

    # field maps
    fig, ax = plt.subplots(nrows=2, ncols=nsteps+2)
    im=ax[0,0].imshow(mesh); fig.colorbar(im, ax=ax[0,0],fraction=0.047); ax[0,0].set_title('geometry')
    ax[0,0].axis('off')
    im=ax[0,1].imshow(stress_in[:,:,0],vmin=0,vmax=20); fig.colorbar(im,ax=ax[0,1],fraction=0.047); ax[0,1].set_title('stress, load'+str(load[0]))
    ax[0,1].axis('off')
    im=ax[1,1].imshow(damage_in,vmin=0,vmax=1); fig.colorbar(im,ax=ax[1,1],fraction=0.047); ax[1,1].set_title('damage, load'+str(load[0]))
    ax[1,1].axis('off')
    for i in range(nsteps):
        im=ax[0,i+2].imshow(stress_out[i][:,:,0],vmin=0); fig.colorbar(im,ax=ax[0,i+2],fraction=0.047); ax[0,i+2].set_title('stress, load'+str(load[i+1]))
        ax[0,i+2].axis('off')
        im=ax[1,i+2].imshow(damage_out[i],vmin=0); fig.colorbar(im,ax=ax[1,i+2],fraction=0.047); ax[1,i+2].set_title('damage, load'+str(load[i+1]))
        ax[1, i + 2].axis('off')

    # macro stress-strain curve (if present)
    if curve_macro is not None:
        ax[1,0].plot(curve_macro[:,0],curve_macro[:,1])
        for i in range(len(load)):
            ax[1,0].plot(load[i], curve_macro[:,1][curve_macro[:,0]==load[i]],'or')
    else:
        ax[1,0].axis('off')

    # fig.subplots_adjust(wspace=0.1)



def imshow_sequence(input, output, curve_macro=None):
    mesh = input[0][0]
    load = input[1].numpy()
    stress_out = output[:,0]
    damage_out = output[:,3]
    nsteps = stress_out.shape[0]

    # field maps
    fig, ax = plt.subplots(nrows=2, ncols=nsteps+1)
    im=ax[0,0].imshow(mesh); fig.colorbar(im, ax=ax[0,0],fraction=0.047); ax[0,0].set_title('geometry')
    ax[0,0].axis('off')
    for i in range(nsteps):
        im=ax[0,i+1].imshow(stress_out[i],vmin=0); fig.colorbar(im,ax=ax[0,i+1],fraction=0.047)
        ax[0,i+1].set_title(str(load[i])); ax[0,i+1].axis('off')
        im=ax[1,i+1].imshow(damage_out[i],vmin=0); fig.colorbar(im,ax=ax[1,i+1],fraction=0.047)
        ax[1,i+1].set_title(str(load[i])); ax[1,i+1].axis('off')

    # macro stress-strain curve (if present)
    if curve_macro is not None:
        ax[1,0].plot(curve_macro[:,0],curve_macro[:,1])
        for i in range(len(load)):
            id = np.abs(curve_macro[:,0]-load[i]) < 1e-9
            ax[1,0].plot(load[i], curve_macro[:,1][id],'or')
    else:
        ax[1,0].axis('off')

    # fig.subplots_adjust(wspace=0.1)

def imshow_sequence1(input, output, curve_macro=None):
    mesh = input[0][0]
    load = input[1].numpy()
    field_out = output[:,0]
    nsteps = len(load)

    # field maps
    fig, ax = plt.subplots(nrows=2, ncols=nsteps)
    im=ax[0,0].imshow(mesh); fig.colorbar(im, ax=ax[0,0],fraction=0.047)
    ax[0,0].set_title('geometry');  ax[0,0].axis('off')
    for i in range(nsteps):
        im=ax[1,i].imshow(field_out[i],vmin=0); fig.colorbar(im,ax=ax[1,i],fraction=0.047)
        ax[1,i].set_title(str(load[i])); ax[1,i].axis('off')

    # macro stress-strain curve (if present)
    if curve_macro is not None:
        ax[0,1].plot(curve_macro[:,0],curve_macro[:,1])
        for i in range(len(load)):
            id = np.abs(curve_macro[:,0]-load[i]) < 1e-9
            ax[0,1].plot(load[i], curve_macro[:,1][id],'or')


import math
def imshow_video(damage):
    nsteps = damage.shape[0]

    # field maps
    ncols = 5
    nrows = math.ceil(nsteps/ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    if nrows==1:
        for i in range(nsteps):
            im = ax[i].imshow(damage[i][0], vmin=0)
            fig.colorbar(im, ax=ax[i], fraction=0.047)
            ax[i].axis('off')

        for i in range(nsteps, nrows * ncols):
            ax[i].axis('off')
    else:
        for i in range(nsteps):
            irow, icol = i//ncols, i%ncols
            im=ax[irow,icol].imshow(damage[i][0],vmin=0); fig.colorbar(im,ax=ax[irow,icol],fraction=0.047)
            ax[irow,icol].axis('off')

        for i in range(nsteps, nrows*ncols):
            irow, icol = i//ncols, i%ncols
            ax[irow,icol].axis('off')



def imshow_colorbar(img, cmin=None, cmax=None):
    plt.imshow(img)
    plt.colorbar()

    if cmin is not None:
        plt.clim(cmin, cmax)

    plt.show()

    # import matplotlib.pyplot as plt
    #
    # import numpy as np
    #
    # m1 = np.random.rand(3, 3)
    # m2 = np.arange(0, 3 * 3, 1).reshape((3, 3))
    #
    # fig = plt.figure(figsize=(16, 12))
    # ax1 = fig.add_subplot(121)
    # im1 = ax1.imshow(m1, interpolation='None')
    #
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax, orientation='vertical')
    #
    # ax2 = fig.add_subplot(122)
    # im2 = ax2.imshow(m2, interpolation='None')
    #
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im2, cax=cax, orientation='vertical')



###
def extractLossEpoch(df):
    trainLoss = list()
    validLoss = list()
    for data in df.values:
        if data[0] == 'Train loss vs epoch':
            trainLoss.append(data[1])
            
        elif data[0] == 'Valid loss vs epoch':
            validLoss.append(data[1])
            
    return trainLoss, validLoss
    
    

