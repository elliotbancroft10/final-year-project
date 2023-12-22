import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision
import os
import shutil
from utils.IOfcts import *
from tqdm import tqdm

# ========================================
# neural network for full field prediction
# ========================================
# input: 501x501 image

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 7, stride=3, padding=2, padding_mode="circular", bias=False)  # 501 -> 167
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2, padding_mode="circular", bias=False)  # 167 -> 84
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 4, stride=2, padding=2, padding_mode="circular", bias=False)  # 84 -> 43
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode="circular", bias=False)  # 43 -> 22
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 4, stride=2, padding=1, padding_mode="circular", bias=False)  # 22 -> 11
        self.bn5 = nn.BatchNorm2d(128)
        # self.conv6 = nn.Conv2d(128, 16, 1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        # x = self.conv6(x)

        return x

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        # self.t_conv6 = nn.Conv2d(16, 128, 1, stride=1, padding=0)
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

        # x = self.t_conv6(x)
        x = self.relu(self.bn4(self.t_conv5(x)))
        x = self.relu(self.bn3(self.t_conv4(x)))
        x = self.relu(self.bn2(self.t_conv3(x)))
        x = self.relu(self.bn1(self.t_conv2(x)))
        x = self.t_conv1(x)

        return x

class fieldPredictor(nn.Module):
    def __init__(self):
        super(fieldPredictor, self).__init__()

        # self.t_conv6 = nn.Conv2d(16, 128, 1, stride=1, padding=0)
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

        # x = self.t_conv6(x)
        x = self.relu(self.bn4(self.t_conv5(x)))
        x = self.relu(self.bn3(self.t_conv4(x)))
        x = self.relu(self.bn2(self.t_conv3(x)))
        x = self.relu(self.bn1(self.t_conv2(x)))
        x = self.t_conv1(x)

        return x


# class fieldPredictor(nn.Module): #didn't work --> too much padding
#     def __init__(self):
#         super(fieldPredictor, self).__init__()
#
#         self.t_conv8 = nn.Conv2d(16, 32, 1, stride=1, padding=0)
#         self.t_conv7 = nn.Conv2d(32, 64, 1, stride=1, padding=0)
#         self.t_conv6 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
#         self.t_conv5 = nn.ConvTranspose2d(128, 96, 3, stride=2, padding=1)  # 11 -> 21
#         self.t_conv4 = nn.ConvTranspose2d(96, 64, 3, stride=2, padding=1) # 21 -> 41
#         self.t_conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1) # 41 -> 81
#         self.t_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1) # 81 -> 161
#         self.t_conv1 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1) # 161 -> 321
#         self.t_conv0 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1)  # 321 -> 641
#         self.conv1 = nn.Conv2d(1, 1, 3, stride=2, padding=181, padding_mode="circular") # 641 -> 501
#         self.bn8 = nn.BatchNorm2d(32)
#         self.bn7 = nn.BatchNorm2d(64)
#         self.bn6 = nn.BatchNorm2d(128)
#         self.bn5 = nn.BatchNorm2d(96)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.bn1 = nn.BatchNorm2d(8)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#
#         x = self.relu(self.bn8(self.t_conv8(x)))
#         x = self.relu(self.bn7(self.t_conv7(x)))
#         x = self.relu(self.bn6(self.t_conv6(x)))
#         x = self.relu(self.bn5(self.t_conv5(x)))
#         x = self.relu(self.bn4(self.t_conv4(x)))
#         x = self.relu(self.bn3(self.t_conv3(x)))
#         x = self.relu(self.bn2(self.t_conv2(x)))
#         x = self.relu(self.bn1(self.t_conv1(x)))
#         x = self.relu(self.t_conv0(x))
#         x = self.conv1(x)
#
#         return x


# ============
# fullfieldNet
# ============
class fullFieldNet(nn.Module):
    def __init__(self):
        super(fullFieldNet, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()
        self.fieldPredictor = fieldPredictor()

    def forward(self, x):
        x = self.encoder(x)
        y = self.fieldPredictor(x) #recon_output
        x = self.decoder(x) #recon_mesh

        return x, y


# def test():
#     batch_size = 64
#     img_channels = 1
#     img_H, img_W = 501, 501
#     net = fullFieldNet()
#     x = torch.randn(batch_size, img_channels, img_H, img_W)
#     y, x = net(x)
#     y.to('cuda')
#     x.to('cuda')
#     print(y.shape)
#     print(x.shape)
#
# test()


# ===============================
# training for the full-field net
# ===============================

def train(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
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
        net.train()
        for batch_idx, (mesh, output) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                mesh = Variable(mesh).cuda()
                output = Variable(output).cuda()

            # forward pass: compute the output (reconsruction)
            recon_mesh, recon_output = net(mesh)

            # calculate the batch loss
            loss = reconLoss(recon_mesh, mesh) + reconLoss(recon_output, output) * 3

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
        net.eval()
        with torch.no_grad():
            for batch_idx, (mesh, output) in enumerate(tqdm(loaders['test']), 0):
                # move to GPU
                if use_cuda:
                    mesh = Variable(mesh).cuda()
                    output = Variable(output).cuda()

                # forward pass: compute the output (reconsruction)
                recon_mesh, recon_output = net(mesh)

                # calculate the batch loss
                loss =  reconLoss(recon_mesh, mesh) + reconLoss(recon_output, output)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

            if dir_tb is not None:
                img_grid = torchvision.utils.make_grid(torch.cat((output[:8], recon_output[:8])))
                writer.add_image('input v.s. output', img_grid)

        # # calculate average losses
        # train_loss = train_loss / len(loaders['train'].dataset)
        # valid_loss = valid_loss / len(loaders['test'].dataset)

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
        writer.add_graph(net, mesh)
        writer.close()

    # return trained model
    return net

