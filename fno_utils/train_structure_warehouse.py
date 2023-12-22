import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision
import os
import shutil
from yc_utils.IOfcts import *
from tqdm import tqdm


# ===============================================================
# training for the full-field net: elastic stress/strain -> crack
# ===============================================================

def train_elastic2crack(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None):

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    for epoch in range(init_epoch, num_epochs+1):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        # net.train()
        for batch_idx, data in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                inputs = Variable(data[0]).cuda()
                outputs = Variable(data[1]).cuda()

            # forward pass: compute the output (reconsruction)
            recon = net(inputs)

            # calculate the batch loss
            loss = reconLoss(recon, outputs)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1


        # ################ #
        # validate the net #
        # ################ #
        with torch.no_grad():
            # net.eval()

            for batch_idx, data in enumerate(tqdm(loaders['test']), 0):

                # move to GPU
                if use_cuda:
                    inputs = Variable(data[0]).cuda()
                    outputs = Variable(data[1]).cuda()

                # forward pass: compute the output (reconsruction)
                recon = net(inputs)

                # calculate the batch loss
                loss = reconLoss(recon, outputs)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # draw the schematic to tensorboard
    if dir_tb is not None:
        writer.add_graph(net, inputs)
        writer.close()

    # return trained model
    return net




# =======================================================
# training for the incremental full-field prediction net
# =======================================================

def train_incremental(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None):

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger - overfit deliberately
    # input, output = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs+1):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (input, output) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                for i in range(len(input)):
                    input[i] = input[i].cuda().float()
                for i in range(len(output)):
                    output[i] = output[i].cuda().float()

            # forward pass: compute the output (reconsruction)
            stress, damage = net(input[0], input[2], input[3], input[1]) #mesh, stress, damage, load

            # calculate the batch loss
            loss = reconLoss['stress'](stress, output[0])/output[0].abs().max() + \
                   reconLoss['damage'](damage, output[1])/output[1].abs().max()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # print('epoch'+str(epoch)+', loss:'+str(loss.item()))

        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (input, output) in enumerate(tqdm(loaders['test']), 0):

                # move to GPU
                if use_cuda:
                    for i in range(len(input)):
                        input[i] = input[i].cuda().float()
                    for i in range(len(output)):
                        output[i] = output[i].cuda().float()

                # forward pass: compute the output (reconsruction)
                stress, damage = net(input[0], input[2], input[3], input[1])  # mesh, stress, damage, load

                # calculate the batch loss
                loss = reconLoss['stress'](stress, output[0])/output[0].abs().max() + \
                       reconLoss['damage'](damage, output[1])/output[1].abs().max()

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net





# ===============================================
# training for the LSTM full-field prediction net
# ===============================================
from torch.nn.utils.rnn import pack_padded_sequence
def train_seqincr(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None):

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0


    # #debugger - overfit deliberately
    # input, output = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs+1):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, load, output, output_lens, _) in enumerate(tqdm(loaders['train']), 0):

            batch_size, _, height, width = mesh.size()
            seq_len = output.shape[1]

            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float()
                tmp = list()
                for i in range(batch_size):
                    tmp.append( load[i].cuda().float() )
                load = tmp
                output = output.cuda().float()

            # forward pass: compute the output (reconsruction)
            predict = net(mesh, load, seq_len, output_lens) #mesh, load

            # calculate the batch loss
            predict = pack_padded_sequence(predict, output_lens, batch_first=True, enforce_sorted=False)
            output = pack_padded_sequence(output, output_lens, batch_first=True, enforce_sorted=False)
            # loss = reconLoss(predict.data[batch_size*2-1:], output.data[batch_size*2-1:])
            loss = reconLoss(predict.data, output.data)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # print('epoch'+str(epoch)+', loss:'+str(loss.item()))

        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, load, output, output_lens, _) in enumerate(tqdm(loaders['test']), 0):

                batch_size, _, height, width = mesh.size()
                seq_len = output.shape[1]

                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float()
                    tmp = list()
                    for i in range(batch_size):
                        tmp.append(load[i].cuda().float())
                    load = tmp
                    output = output.cuda().float()

                # forward pass: compute the output (reconsruction)
                predict = net(mesh, load, seq_len, output_lens)  # mesh, load

                # calculate the batch loss
                predict = pack_padded_sequence(predict, output_lens, batch_first=True, enforce_sorted=False)
                output = pack_padded_sequence(output, output_lens, batch_first=True, enforce_sorted=False)
                # loss = reconLoss(predict.data[batch_size*2-1:], output.data[batch_size*2-1:])
                loss = reconLoss(predict.data, output.data)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net







# ==============================================================================
# training for the LSTM full-field prediction net, pure video prediction method
# ==============================================================================
def train_videopred(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None):

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0


    # #debugger - overfit deliberately
    # input, output = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs+1):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, output, seq_lens) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float()
                output = output.cuda().float()


            # forward pass: compute the output (reconsruction)
            pred = net(output[:, :3], 7)
            # calculate the batch loss
            loss = reconLoss(pred, output[:, 3:10])

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # print('epoch'+str(epoch)+', loss:'+str(loss.item()))

        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, output, seq_lens) in enumerate(tqdm(loaders['test']), 0):

                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float()
                    output = output.cuda().float()

                # forward pass: compute the output (reconsruction)
                pred = net(output[:, :3], 7)
                # calculate the batch loss
                loss = reconLoss(pred, output[:, 3:10])

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net



# ========================================================================
# training for the full-field prediction net for damage at individual step
# ========================================================================

def train_stepdamage(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None):

    #                   lambda,   mu,       Gc, lc
    mParam = np.array([[ 4.1667e3,  6.2500e3,     1, 0.5e-3],    #fibre
                       [ 4.7953e3,  1.3525e3, 0.011, 0.5e-3],    #matrix
                       [4.7953e-2, 1.3525e-2,    0., 0.5e-3]     #pore
                       ])
    nc = mParam.shape[1]

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0


    # #debugger - overfit deliberately
    # input, output = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs+1):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, load, field) in enumerate(tqdm(loaders['train']), 0):
            nb, _, h, w = mesh.shape
            mParamMAP = torch.zeros(nc, nb, h, w)

            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float()
                load = load.cuda().float()
                field = field.cuda().float()
                mParamMAP = mParamMAP.cuda().float()

            # prepare initial mesh - mParam
            for ic in range(nc):
                for im in range(3):
                    mParamMAP[ic, mesh[:,0]==im+1] = mParam[im, ic]
            mParamMAP = torch.moveaxis(mParamMAP, 1, 0)

            # forward pass: compute the output
            pred = net(mParamMAP, torch.unsqueeze(load[:,0], 1))

            # calculate the batch loss
            loss = reconLoss(pred, field)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # print('epoch'+str(epoch)+', loss:'+str(loss.item()))

        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, load, field) in enumerate(tqdm(loaders['test']), 0):
                nb, _, h, w = mesh.shape
                mParamMAP = torch.zeros(nc, nb, h, w)

                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float()
                    load = load.cuda().float()
                    field = field.cuda().float()
                    mParamMAP = mParamMAP.cuda().float()

                # prepare initial mesh - mParam
                for ic in range(nc):
                    for im in range(3):
                        mParamMAP[ic, mesh[:, 0] == im + 1] = mParam[im, ic]
                mParamMAP = torch.moveaxis(mParamMAP, 1, 0)

                # forward pass: compute the output
                pred = net(mParamMAP, torch.unsqueeze(load[:,0], 1))

                # calculate the batch loss
                loss = reconLoss(pred, field)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net


# =====================================================
def lossPhaseField(damage, strain, mParamMAP):
    filter = torch.tensor([[0.,  1., 0.],
                           [1., -4., 1.],
                           [0.,  1., 0.]])
    filter = filter.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

    if damage.is_cuda:
        filter = filter.cuda().float()

    laplacianD = torch.nn.functional.conv2d(damage, filter, padding=1)
    energy = 0.5 * mParamMAP[:,0] * (strain[:,0]+strain[:,1])**2 + \
             mParamMAP[:,1] * (strain[:,0]**2 + strain[:,1]**2 + 0.5*strain[:,2]**2)

    residual =  mParamMAP[:,2] * mParamMAP[:,3]**2 * laplacianD[:,0] + \
                2. * mParamMAP[:,3] * energy - ( 2. * mParamMAP[:,3] * energy + mParamMAP[:, 2]) * damage[:,0]
    return torch.mean(residual**2)

def lossPhaseField_Energy(damage, strain, mParamMAP):

    # vxsiz = 2e-4  #mm

    # deriv5pt = torch.tensor([[1., -8., 0., 8., -1.]])
    # filter_x = deriv5pt.view(1, 1, 1, 5).repeat(1, 1, 1, 1)
    # filter_y = deriv5pt.view(1, 1, 5, 1).repeat(1, 1, 1, 1)
    deriv5pt = torch.tensor([[-1., 0., 1.]])
    filter_x = deriv5pt.view(1, 1, 1, 3).repeat(1, 1, 1, 1)
    filter_y = deriv5pt.view(1, 1, 3, 1).repeat(1, 1, 1, 1)

    if damage.is_cuda:
        filter_x = filter_x.cuda().float()
        filter_y = filter_y.cuda().float()

    gradientD2 = torch.nn.functional.conv2d(damage, filter_x, padding=(0,1))**2 + \
                 torch.nn.functional.conv2d(damage, filter_y, padding=(1,0))**2
    energyE = ( 0.5 * mParamMAP[:,0] * (strain[:,0]+strain[:,1])**2 + \
                mParamMAP[:,1] * (strain[:,0]**2 + strain[:,1]**2 + 0.5*strain[:,2]**2) ) * (1-damage[:,0])**2

    energyD = mParamMAP[:,2] * (0.5/mParamMAP[:,3] * damage[:,0]**2 + 0.5*mParamMAP[:,3]*gradientD2[:,0])

    return torch.mean(energyE + energyD)



def train_stepdamage_withMech(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None, varLOAD=False):

    #                      lambda,        mu,    Gc,     lc
    mParam = np.array([[4.1667e3,  6.2500e3,   0.1, 0.5e-3],    #fibre
                       [4.7953e3,  1.3525e3, 0.011, 0.5e-3],    #matrix
                       [    1e-3,      1e-3,  0.11, 0.5e-3]     #pore
                       ])
    nc = mParam.shape[1]

    # normalise the parameters
    for i in range(mParam.shape[1]):
        mParam[:,i] = mParam[:,i] / mParam[:,i].max()

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger
    # (mesh, load, strain) = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, load, strain) in enumerate(tqdm(loaders['train']), 0):
            nb, _, h, w = mesh.shape
            mParamMAP = torch.zeros(nc, nb, h, w)

            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float()
                load = load.cuda().float()
                strain = strain.cuda().float()
                mParamMAP = mParamMAP.cuda().float()

            # prepare initial mesh - mParam
            for ic in range(nc):
                for im in range(3):
                    mParamMAP[ic, mesh[:,0]==im+1] = mParam[im, ic]
            mParamMAP = torch.moveaxis(mParamMAP, 1, 0)

            # forward pass: compute the output
            if varLOAD==True:
                pred = net(mParamMAP[:,:2], torch.unsqueeze(load[:,0], 1))
            elif varLOAD==False:
                pred = net(mParamMAP[:,:2])

            # calculate the batch loss
            loss = lossPhaseField_Energy(pred, strain, mParamMAP)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # net.eval()
            # with torch.no_grad():
            #     # pred = net(mParamMAP[:, :2], torch.unsqueeze(load[:, 0], 1))
            #     pred = net(mParamMAP[:, :2])
            #     loss1 = lossPhaseField_Energy(pred, strain, mParamMAP)
            # print('epoch'+str(epoch)+', train loss:'+str(loss.item())+', eval loss:'+str(loss1.item()))


        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, load, strain) in enumerate(tqdm(loaders['test']), 0):
                nb, _, h, w = mesh.shape
                mParamMAP = torch.zeros(nc, nb, h, w)

                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float()
                    load = load.cuda().float()
                    strain = strain.cuda().float()
                    mParamMAP = mParamMAP.cuda().float()

                # prepare initial mesh - mParam
                for ic in range(nc):
                    for im in range(3):
                        mParamMAP[ic, mesh[:, 0] == im + 1] = mParam[im, ic]
                mParamMAP = torch.moveaxis(mParamMAP, 1, 0)

                # forward pass: compute the output
                if varLOAD==True:
                    pred = net(mParamMAP[:,:2], torch.unsqueeze(load[:,0], 1))
                elif varLOAD==False:
                    pred = net(mParamMAP[:,:2])

                # calculate the batch loss
                loss = lossPhaseField(pred, strain, mParamMAP)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net
    
    
    
    
    

#####################

def train_mesh2stress(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None, varLOAD=False, num_field=1):

    #                      lambda,        mu,    Gc,     lc
    mParam = np.array([[4.1667e3,  6.2500e3,   0.1, 0.5e-3],    #fibre
                       [4.7953e3,  1.3525e3, 0.011, 0.5e-3],    #matrix
                       [    1e-3,      1e-3,  0.11, 0.5e-3]     #pore
                       ])
    nc = mParam.shape[1]
    
    # normalise the parameters
    for i in range(mParam.shape[1]):
        mParam[:,i] = mParam[:,i] / mParam[:,i].max()
    
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger
    # (mesh, load, strain) = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, load, stress) in enumerate(tqdm(loaders['train']), 0):
            nb, _, h, w = mesh.shape
            mParamMAP = torch.zeros(nc, nb, h, w)

            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float()
                load = load.cuda().float()
                stress = stress.cuda().float()
                mParamMAP = mParamMAP.cuda().float()

            # prepare initial mesh - mParam
            for ic in range(nc):
                for im in range(3):
                    mParamMAP[ic, mesh[:,0]==im+1] = mParam[im, ic]
            mParamMAP = torch.moveaxis(mParamMAP, 1, 0)

            # forward pass: compute the output
            pred = net(mParamMAP[:,:2])
            #pred = net(mesh/3)

            # calculate the batch loss
            if num_field == 1:
                loss = reconLoss(pred, torch.unsqueeze(stress[:,0], 1))
            elif num_field == 3:
                loss = reconLoss(pred, stress)
                
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # net.eval()
            # with torch.no_grad():
            #     # pred = net(mParamMAP[:, :2], torch.unsqueeze(load[:, 0], 1))
            #     pred = net(mParamMAP[:, :2])
            #     loss1 = lossPhaseField_Energy(pred, strain, mParamMAP)
            # print('epoch'+str(epoch)+', train loss:'+str(loss.item())+', eval loss:'+str(loss1.item()))

        if dir_tb is not None:
            writer.add_scalar('Train loss vs epoch', train_loss, global_step=epoch)
        
        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, load, stress) in enumerate(tqdm(loaders['test']), 0):
                nb, _, h, w = mesh.shape
                mParamMAP = torch.zeros(nc, nb, h, w)

                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float()
                    load = load.cuda().float()
                    stress = stress.cuda().float()
                    mParamMAP = mParamMAP.cuda().float()

                # prepare initial mesh - mParam
                for ic in range(nc):
                    for im in range(3):
                        mParamMAP[ic, mesh[:, 0] == im + 1] = mParam[im, ic]
                mParamMAP = torch.moveaxis(mParamMAP, 1, 0)

                # forward pass: compute the output
                pred = net(mParamMAP[:,:2])
                #pred = net(mesh/3.)

                # calculate the batch loss
                if num_field == 1:
                    loss = reconLoss(pred, torch.unsqueeze(stress[:,0], 1))
                elif num_field == 3:
                    loss = reconLoss(pred, stress)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1
            
            if dir_tb is not None:
                writer.add_scalar('Valid loss vs epoch', valid_loss, global_step=epoch)

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net



def train_mesh2stress_2(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None, varLOAD=False, num_field=1):
    
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger
    # (mesh, load, strain) = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, load, stress) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float() / 3.
                load = load.cuda().float()
                stress = stress.cuda().float()

            # forward pass: compute the output
            pred = net(mesh)

            # calculate the batch loss
            if num_field == 1:
                loss = reconLoss(pred, torch.unsqueeze(stress[:,0], 1))
            elif num_field == 3:
                loss = reconLoss(pred, stress)
                
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # net.eval()
            # with torch.no_grad():
            #     # pred = net(mParamMAP[:, :2], torch.unsqueeze(load[:, 0], 1))
            #     pred = net(mParamMAP[:, :2])
            #     loss1 = lossPhaseField_Energy(pred, strain, mParamMAP)
            # print('epoch'+str(epoch)+', train loss:'+str(loss.item())+', eval loss:'+str(loss1.item()))

        if dir_tb is not None:
            writer.add_scalar('Train loss vs epoch', train_loss, global_step=epoch)
        
        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, load, stress) in enumerate(tqdm(loaders['test']), 0):
                
                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float() / 3.
                    load = load.cuda().float()
                    stress = stress.cuda().float()

                # forward pass: compute the output
                pred = net(mesh)

                # calculate the batch loss
                if num_field == 1:
                    loss = reconLoss(pred, torch.unsqueeze(stress[:,0], 1))
                elif num_field == 3:
                    loss = reconLoss(pred, stress)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1
            
            if dir_tb is not None:
                writer.add_scalar('Valid loss vs epoch', valid_loss, global_step=epoch)

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net

def train_mesh2damage_2(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None, varLOAD=False, num_field=1):
    
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger
    # (mesh, load, strain) = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, load, damage) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float() / 3.
                load = load.cuda().float()
                damage = damage.cuda().float()

            # forward pass: compute the output
            pred = net(mesh)

            # calculate the batch loss
            if num_field == 1:
                loss = reconLoss(pred, torch.unsqueeze(damage[:,0], 1))
            elif num_field == 3:
                loss = reconLoss(pred, damage)
                
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # net.eval()
            # with torch.no_grad():
            #     # pred = net(mParamMAP[:, :2], torch.unsqueeze(load[:, 0], 1))
            #     pred = net(mParamMAP[:, :2])
            #     loss1 = lossPhaseField_Energy(pred, strain, mParamMAP)
            # print('epoch'+str(epoch)+', train loss:'+str(loss.item())+', eval loss:'+str(loss1.item()))

        if dir_tb is not None:
            writer.add_scalar('Train loss vs epoch', train_loss, global_step=epoch)
        
        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, load, damage) in enumerate(tqdm(loaders['test']), 0):
                
                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float() / 3.
                    load = load.cuda().float()
                    damage = damage.cuda().float()

                # forward pass: compute the output
                pred = net(mesh)

                # calculate the batch loss
                if num_field == 1:
                    loss = reconLoss(pred, torch.unsqueeze(damage[:,0], 1))
                elif num_field == 3:
                    loss = reconLoss(pred, damage)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1
            
            if dir_tb is not None:
                writer.add_scalar('Valid loss vs epoch', valid_loss, global_step=epoch)

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net



def MSELoss_p(pred, field, p=1.):

    return torch.mean((pred - field)**2 * (1.+field*p)) 


def train_elas2crk(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None, varLOAD=False):


    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger
    # (mesh, load, strain) = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (_, energy, load, field) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                load = load.cuda().float()
                energy = energy.cuda().float()
                field = field.cuda().float()


            # forward pass: compute the output
            pred = net(energy)

            # calculate the batch loss
            loss = reconLoss(pred, field)
            #loss = MSELoss_p(pred, field, 2)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # net.eval()
            # with torch.no_grad():
            #     # pred = net(mParamMAP[:, :2], torch.unsqueeze(load[:, 0], 1))
            #     pred = net(mParamMAP[:, :2])
            #     loss1 = lossPhaseField_Energy(pred, strain, mParamMAP)
            # print('epoch'+str(epoch)+', train loss:'+str(loss.item())+', eval loss:'+str(loss1.item()))
        
        if dir_tb is not None:
            writer.add_scalar('Train loss vs epoch', train_loss, global_step=epoch)
        

        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (_, energy, load, field) in enumerate(tqdm(loaders['test']), 0):

                # move to GPU
                if use_cuda:
                    energy = energy.cuda().float()
                    load = load.cuda().float()
                    field = field.cuda().float()


                # forward pass: compute the output
                pred = net(energy)

                # calculate the batch loss
                loss = reconLoss(pred, field)
                #loss = MSELoss_p(pred, field, 2)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1
                    
            if dir_tb is not None:
                writer.add_scalar('Valid loss vs epoch', valid_loss, global_step=epoch)

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net
    
    
    

def train_elas2crk_PINN(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None):

    #                      lambda,        mu,    Gc,     lc
    mParam = np.array([[4.1667e3,  6.2500e3,   0.1, 0.5e-3],    #fibre
                       [4.7953e3,  1.3525e3, 0.011, 0.5e-3],    #matrix
                       [    1e-3,      1e-3,  0.11, 0.5e-3]     #pore
                       ])
    nc = mParam.shape[1]

    # normalise the parameters
    for i in range(mParam.shape[1]):
        mParam[:,i] = mParam[:,i] / mParam[:,i].max()

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger
    # (mesh, load, strain) = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, energy, load, field) in enumerate(tqdm(loaders['train']), 0):
            nb, _, h, w = mesh.shape
            mParamMAP = torch.zeros(nc, nb, h, w)

            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float()
                energy = energy.cuda().float()
                load = load.cuda().float()
                field = field.cuda().float()
                mParamMAP = mParamMAP.cuda().float()

            # prepare initial mesh - mParam
            for ic in range(nc):
                for im in range(3):
                    mParamMAP[ic, mesh[:,0]==im+1] = mParam[im, ic]
            mParamMAP = torch.moveaxis(mParamMAP, 1, 0)

            # forward pass: compute the output
            pred = net(energy)

            # calculate the batch loss
            loss = lossPhaseField_Energy(pred, field[:,1:], mParamMAP)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # net.eval()
            # with torch.no_grad():
            #     # pred = net(mParamMAP[:, :2], torch.unsqueeze(load[:, 0], 1))
            #     pred = net(mParamMAP[:, :2])
            #     loss1 = lossPhaseField_Energy(pred, strain, mParamMAP)
            # print('epoch'+str(epoch)+', train loss:'+str(loss.item())+', eval loss:'+str(loss1.item()))
        
        if dir_tb is not None:
            writer.add_scalar('Train loss vs epoch', train_loss, global_step=epoch)
        
        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, energy, load, field) in enumerate(tqdm(loaders['test']), 0):
                nb, _, h, w = mesh.shape
                mParamMAP = torch.zeros(nc, nb, h, w)

                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float()
                    energy = energy.cuda().float()
                    load = load.cuda().float()
                    field = field.cuda().float()
                    mParamMAP = mParamMAP.cuda().float()

                # prepare initial mesh - mParam
                for ic in range(nc):
                    for im in range(3):
                        mParamMAP[ic, mesh[:, 0] == im + 1] = mParam[im, ic]
                mParamMAP = torch.moveaxis(mParamMAP, 1, 0)


                # forward pass: compute the output
                pred = net(energy)

                # calculate the batch loss
                loss = lossPhaseField(pred, field[:,1:], mParamMAP)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

            if dir_tb is not None:
                writer.add_scalar('Valid loss vs epoch', valid_loss, global_step=epoch)

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net
    
    
    
################### macro-curve #####################
def MSELoss_stressANDstrain(pred, curve):
    loss = ((pred[:,0] - curve[:,0]) / curve[:,0].max())**2 + \
           ((pred[:,1] - curve[:,1]) / curve[:,1].max())**2
    return torch.mean(loss) 


def train_mesh2curve(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None):


    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger
    # (mesh, load, strain) = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, curve) in enumerate(tqdm(loaders['train']), 0):
            
            nb, _, seq_len = curve.size()
            
            # incremental load steps (to be scaled)
            incrs = torch.unsqueeze(torch.linspace(0., 1., seq_len), 0).repeat(nb, 1)
            
            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float()
                curve = curve.cuda().float()
                incrs = incrs.cuda().float()

            # forward pass: compute the output
            pred = net(mesh, incrs)

            # calculate the batch loss
            loss = MSELoss_stressANDstrain(pred, curve)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # net.eval()
            # with torch.no_grad():
            #     # pred = net(mParamMAP[:, :2], torch.unsqueeze(load[:, 0], 1))
            #     pred = net(mParamMAP[:, :2])
            #     loss1 = lossPhaseField_Energy(pred, strain, mParamMAP)
            # print('epoch'+str(epoch)+', train loss:'+str(loss.item())+', eval loss:'+str(loss1.item()))
        
        if dir_tb is not None:
            writer.add_scalar('Train loss vs epoch', train_loss, global_step=epoch)
        

        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, curve) in enumerate(tqdm(loaders['test']), 0):
            
                nb, _, seq_len = curve.size()
            
                # incremental load steps (to be scaled)
                incrs = torch.unsqueeze(torch.linspace(0., 1., seq_len), 0).repeat(nb, 1)
            
                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float()
                    curve = curve.cuda().float()
                    incrs = incrs.cuda().float()

                # forward pass: compute the output
                pred = net(mesh, incrs)

                # calculate the batch loss
                loss = MSELoss_stressANDstrain(pred, curve)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1
                    
            if dir_tb is not None:
                writer.add_scalar('Valid loss vs epoch', valid_loss, global_step=epoch)

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net
    
    
    
