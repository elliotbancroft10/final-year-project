import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision
import os
import shutil
from yc_utils.IOfcts import *
from tqdm import tqdm
from yc_utils.NN_autoencoder import autoencoderNet


# ================================================================
# neural network for latent layers (input-latent -> output-latent)
# ================================================================

class latentNet(nn.Module):
    def __init__(self):
        super(latentNet, self).__init__()

        self.conv1 = nn.Conv2d(16, 128, 3, stride=2, padding=1, padding_mode="circular", bias=True) #11 -> 6
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=0, bias=True) #6 -> 4
        self.conv3 = nn.Conv2d(256, 512, 3, stride=1, padding=0, bias=True) #4 -> 2
        self.conv4 = nn.Conv2d(512, 1024, 2, stride=1, padding=0, bias=True) #2 -> 1
        self.flaten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(1024*1*1, 100)
        self.fc2 = nn.Linear(100, 1024*1*1)
        self.unflaten = nn.Unflatten(dim=1, unflattened_size=(1024, 1, 1))
        self.t_conv4 = nn.ConvTranspose2d(1024, 512, 2, stride=1, padding=0)
        self.t_conv3 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0)
        self.t_conv2 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=0)
        self.t_conv1 = nn.ConvTranspose2d(128, 16, 3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flaten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.unflaten(x)
        x = self.relu(self.t_conv4(x))
        x = self.relu(self.t_conv3(x))
        x = self.relu(self.t_conv2(x))
        x = self.relu(self.t_conv1(x))

        return x


class fullfieldNet_latent(nn.Module):
    def __init__(self, AE_input, AE_output, latentPredict):
        super(fullfieldNet_latent, self).__init__()

        # autoencoder for input / output
        AE_input.eval()
        AE_output.eval()
        self.AE_input = AE_input
        self.AE_output = AE_output
        self.latentPredict = latentPredict

        # delete the encoder / decoder part to avoid useless calculation
        AE_input.decoder = nn.Identity()
        AE_output.encoder = nn.Identity()

        # deactivate gradient calculations
        for param in AE_input.parameters():
            param.requires_grad = False

        for param in AE_output.parameters():
            param.requires_grad = False

        for param in latentPredict.parameters():
            param.requires_grad = False

    def forward(self, x):
        _, x = self.AE_input(x)
        x = self.latentPredict(x)
        x, _ = self.AE_output(x)

        return x


def test():
    batch_size = 64
    img_channels = 1
    img_H, img_W = 501, 501
    AE_input = autoencoderNet()
    AE_output = autoencoderNet()
    latent = latentNet()
    net = fullfieldNet_latent(AE_input, AE_output, latent)
    x = torch.randn(batch_size, img_channels, img_H, img_W)
    y = net(x)
    #y.to('cuda')
    print(y.shape)

test()



# ===============================
# training for the full-field net
# ===============================

def train_latent(AE_input, AE_output, latentnet, init_epoch, num_epochs, valid_loss_min_input, loaders,
                 scheduler, optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None):
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    for epoch in range(init_epoch, num_epochs + 1):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        for batch_idx, (mesh, output) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                mesh = Variable(mesh).cuda()
                output = Variable(output).cuda()

            # prepare latent data for training
            with torch.no_grad():
                _, latent_in = AE_input(mesh)
                _, latent_out = AE_output(output)

            # prediction of latent layers
            latent_pred = latentnet(latent_in)

            # calculate the batch loss
            loss = reconLoss(latent_pred, latent_out)

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
        # latentNet.eval()
        with torch.no_grad():
            for batch_idx, (mesh, output) in enumerate(tqdm(loaders['test']), 0):
                # move to GPU
                if use_cuda:
                    mesh = Variable(mesh).cuda()
                    output = Variable(output).cuda()

                # prepare the latents
                with torch.no_grad():
                    _, latent_in = AE_input(mesh)
                    _, latent_out = AE_output(output)

                # prediction
                latent_pred = latentnet(latent_in)

                # calculate the batch loss
                loss =  reconLoss(latent_pred, latent_out)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

        # update learning rate
        scheduler.step()

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': latentnet.state_dict(),
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
    return latentnet

