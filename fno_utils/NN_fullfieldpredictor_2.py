import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision
import os
import shutil
from utils.IOfcts import *
from tqdm import tqdm
from utils.NN_autoencoder import autoencoderNet

# ========================================
# neural network for full field prediction
# ========================================
# input: 501x501 image

class fieldPredictor(nn.Module):
    def __init__(self):
        super(fieldPredictor, self).__init__()

        self.t_conv6 = nn.Conv2d(16, 128, 1, stride=1, padding=0)
        self.t_conv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 11 -> 22
        self.t_conv4 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1) # 22 -> 43
        self.t_conv3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=2) # 43 -> 84
        self.t_conv2 = nn.ConvTranspose2d(16, 8, 5, stride=2, padding=2) # 84 -> 167
        self.t_conv1 = nn.ConvTranspose2d(8, 1, 7, stride=3, padding=2) # 167 -> 501
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.t_conv6(x)
        x = self.relu(self.bn4(self.t_conv5(x)))
        x = self.relu(self.bn3(self.t_conv4(x)))
        x = self.relu(self.bn2(self.t_conv3(x)))
        x = self.relu(self.bn1(self.t_conv2(x)))
        x = self.t_conv1(x)

        return x


# ============
# fullfieldNet
# ============
class fullFieldNet(nn.Module):
    def __init__(self, AE_weights_pth):
        super(fullFieldNet, self).__init__()

        # initialize AutoEncoder
        self.autoencoder = autoencoderNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder.to(self.device)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters())
        self.autoencoder, _, _, _ = load_ckp(AE_weights_pth, self.autoencoder, self.optimizer)

        # freeze the calcultion of gradients for AE
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # prediction branch
        self.fieldPredictor = fieldPredictor()

    def forward(self, x):
        _, x = self.autoencoder(x)
        x = self.fieldPredictor(x)

        return x


# ===============================
# training for the full-field net
# ===============================

def train(AEnet, predictor, init_epoch, num_epochs, valid_loss_min_input, loaders, scheduler,
          optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None):
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
        AEnet.train()
        predictor.train()
        for batch_idx, (mesh, output) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                mesh = Variable(mesh).cuda()
                output = Variable(output).cuda()

            # forward pass: compute the output (reconsruction)
            with torch.no_grad():
                _, y = AEnet(mesh)
            recon_output = predictor(y)

            # calculate the batch loss
            loss =reconLoss(recon_output, output)

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
        AEnet.train() #--> to make sure that input format is the same for train and valid
        predictor.eval()
        with torch.no_grad():
            for batch_idx, (mesh, output) in enumerate(tqdm(loaders['test']), 0):
                # move to GPU
                if use_cuda:
                    mesh = Variable(mesh).cuda()
                    output = Variable(output).cuda()

                # forward pass: compute the output (reconsruction)
                _, y = AEnet(mesh)
                recon_output = predictor(y)

                # calculate the batch loss
                loss =  reconLoss(recon_output, output)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

            if dir_tb is not None:
                img_grid = torchvision.utils.make_grid(torch.cat((output[:8], recon_output[:8])))
                writer.add_image('input v.s. output', img_grid)

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
            'state_dict': predictor.state_dict(),
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
    return predictor

