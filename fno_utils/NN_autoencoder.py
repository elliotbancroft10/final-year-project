import torch
import torch.nn as nn
from torch.autograd import Variable
from yc_utils.IOfcts import *
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
from tqdm import tqdm

# ===============================
# neural network for auto encoder
# ===============================
# input: 501x501 image

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


# class autoencoderNet(nn.Module):
#     def __init__(self):
#         super(autoencoderNet, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1, padding_mode="circular", bias=False) # 251, 251
#         self.bn1 = nn.BatchNorm2d(8)
#         self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1, padding_mode="circular", bias=False) # 126, 126
#         self.bn2 = nn.BatchNorm2d(16)
#         self.conv3 = nn.Conv2d(16, 32, 4, stride=4, padding=1, padding_mode="circular", bias=False) # 32, 32
#         self.bn3 = nn.BatchNorm2d(32)
#         self.conv4 = nn.Conv2d(32, 64, 4, stride=2, padding=2, padding_mode="circular", bias=False) # 17, 17
#         self.bn4 = nn.BatchNorm2d(64)
#         self.conv5 = nn.Conv2d(64, 16, 1, stride=1, padding=0, padding_mode="circular", bias=False)  # 17, 174
#         self.bn5 = nn.BatchNorm2d(16)
#         self.conv6 = nn.Conv2d(16, 32, 3, stride=2, padding=1, padding_mode="circular", bias=False)  # 9, 9
#         self.bn6 = nn.BatchNorm2d(32)
#         self.conv7 = nn.Conv2d(32, 8, 1, stride=1, padding=0, padding_mode="circular", bias=False)  # 9, 9
#         self.bn7 = nn.BatchNorm2d(8)
#
#         self.t_conv7 = nn.ConvTranspose2d(8, 32, 1, stride=1, padding=0)  # 48, 9, 9
#         self.t_conv6 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)  # 32, 17, 17
#         self.t_conv5 = nn.ConvTranspose2d(16, 64, 1, stride=1, padding=0)  # 128, 17, 17
#         self.t_conv4 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1) # 64, 32, 32
#         self.t_conv3 = nn.ConvTranspose2d(32, 16, 4, stride=4, padding=1) # 32, 126, 126
#         self.t_conv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1) # 16, 251, 251
#         self.t_conv1 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1) # 1, 501, 501
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#         # Initialization
#         nn.init.kaiming_normal_(self.conv7.weight, mode='fan_in',
#                                 nonlinearity='relu')
#         nn.init.kaiming_normal_(self.conv6.weight, mode='fan_in',
#                                 nonlinearity='relu')
#         nn.init.xavier_normal_(self.t_conv1.weight)
#
#     def forward(self, x):
#         # encoder
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = self.relu(self.bn5(self.conv5(x)))
#         x = self.relu(self.bn6(self.conv6(x)))
#         x = self.relu(self.bn7(self.conv7(x)))
#
#         y = self.relu(self.bn6(self.t_conv7(x)))
#         y = self.relu(self.bn5(self.t_conv6(y)))
#         y = self.relu(self.bn4(self.t_conv5(y)))
#         y = self.relu(self.bn3(self.t_conv4(y)))
#         y = self.relu(self.bn2(self.t_conv3(y)))
#         y = self.relu(self.bn1(self.t_conv2(y)))
#         # y = nn.Softmax(dim=1)(self.t_conv1(y))
#         y = self.t_conv1(y)
#
#         return y, x

# ==========================
# ==========================
# ==========================
#
# class block(nn.Module):
#     def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
#         super(block, self).__init__()
#         self.expansion = 4
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
#                                padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
#                                padding_mode="circular", padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1,
#                                stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
#         self.relu = nn.ReLU()
#         self.identity_downsample = identity_downsample
# 
#     def forward(self, x):
#         identity = x
# 
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
# 
#         if self.identity_downsample is not None:
#             identity = self.identity_downsample(identity)
# 
#         x += identity
#         x = self.relu(x)
#         return x
# 
# class ResNet(nn.Module):
#     def __init__(self, block, layers, image_channels, num_classes):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2,
#                                padding=3, padding_mode="circular", bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# 
#         # ResNet layers
#         self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
#         self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
#         self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
#         self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)
#         
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512*4, num_classes)
# 
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
# 
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
# 
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
# 
#         return x
# 
#     def _make_layer(self, block, num_residul_blocks, out_channels, stride):
#         identity_downsampe = None
#         layers = []
# 
#         if stride != 1 or self.in_channels != out_channels*4:
#             identity_downsampe = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4,
#                                                          kernel_size=1, stride=stride),
#                                                nn.BatchNorm2d(out_channels*4))
# 
#         layers.append(block(self.in_channels, out_channels, identity_downsampe, stride))
#         self.in_channels = out_channels*4
# 
#         for i in range(num_residul_blocks-1):
#             layers.append(block(self.in_channels, out_channels))
# 
#         return nn.Sequential(*layers)

# def ResNet50(img_channels=3, num_classes=1000):
#     return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)
#
# def ResNet101(img_channels, num_classes=1000):
#     return ResNet(block, [3, 4, 23, 3], img_channels, num_classes)
#
# def ResNet152(img_channels, num_classes=1000):
#     return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)

# def test():
#     img_channels = 1
#     batch_size = 16
#     img_H, img_W = 501, 501
#     net = ResNet50(img_channels=img_channels, num_classes=1000)
#     x = torch.randn(batch_size, img_channels, img_H, img_W)
#     y = net(x).to('cuda')
#     print(y.shape)
#
# test()

# ================================
# ================================
# ================================
# ================================

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, r=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.se = SE_Block(planes, r)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # add SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(encoder, self).__init__()

        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(self.in_channels, 8, 7, stride=3, padding=2, padding_mode="circular", bias=False) # 501 -> 167
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2, padding_mode="circular", bias=False) # 167 -> 84
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 4, stride=2, padding=2, padding_mode="circular", bias=False) # 84 -> 43
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode="circular", bias=False) # 43 -> 22
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 4, stride=2, padding=1, padding_mode="circular", bias=False) # 22 -> 11
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = conv1x1(128, 16, stride=1) #11x11x16
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        return x


class decoder(nn.Module):

    def __init__(self, out_channels=1):
        super(decoder, self).__init__()

        self.out_channels = out_channels
        self.t_conv6 = conv1x1(16, 128, stride=1) #in:11x11x16 --> ...
        self.t_conv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 11 -> 22
        self.t_conv4 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1) # 22 -> 43
        self.t_conv3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=2) # 43 -> 84
        self.t_conv2 = nn.ConvTranspose2d(16, 8, 5, stride=2, padding=2) # 84 -> 167
        self.t_conv1 = nn.ConvTranspose2d(8, self.out_channels, 7, stride=3, padding=2) # 167 -> 501
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.t_conv6(x)
        x = self.relu(self.bn4(self.t_conv5(x)))
        x = self.relu(self.bn3(self.t_conv4(x)))
        x = self.relu(self.bn2(self.t_conv3(x)))
        x = self.relu(self.bn1(self.t_conv2(x)))
        x = self.t_conv1(x)

        return x

class autoencoderNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(autoencoderNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = encoder(self.in_channels)
        self.decoder = decoder(self.out_channels)

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)

        return y, x

# class autoencoderNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super(autoencoderNet, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv1 = nn.Conv2d(self.in_channels, 8, 7, stride=3, padding=2, padding_mode="circular", bias=False) # 501 -> 167
#         self.bn1 = nn.BatchNorm2d(8)
#         self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2, padding_mode="circular", bias=False) # 167 -> 84
#         self.bn2 = nn.BatchNorm2d(16)
#         self.conv3 = nn.Conv2d(16, 32, 4, stride=2, padding=2, padding_mode="circular", bias=False) # 84 -> 43
#         self.bn3 = nn.BatchNorm2d(32)
#         self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode="circular", bias=False) # 43 -> 22
#         self.bn4 = nn.BatchNorm2d(64)
#         self.conv5 = nn.Conv2d(64, 128, 4, stride=2, padding=1, padding_mode="circular", bias=False) # 22 -> 11
#         self.bn5 = nn.BatchNorm2d(128)
#         self.conv6 = conv1x1(128, 16, stride=1)
#
#         self.t_conv6 = conv1x1(16, 128, stride=1)
#         self.t_conv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 11 -> 22
#         self.t_conv4 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1) # 22 -> 43
#         self.t_conv3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=2) # 43 -> 84
#         self.t_conv2 = nn.ConvTranspose2d(16, 8, 5, stride=2, padding=2) # 84 -> 167
#         self.t_conv1 = nn.ConvTranspose2d(8, self.out_channels, 7, stride=3, padding=2) # 167 -> 501
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)
#         # self.dropout = nn.Dropout2d(0.1)
#
#     def forward(self, x):
#         # encoder
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = self.relu(self.bn5(self.conv5(x)))
#         x = self.conv6(x)
#
#         # decoder
#         y = self.t_conv6(x)
#         y = self.relu(self.bn4(self.t_conv5(y)))
#         y = self.relu(self.bn3(self.t_conv4(y)))
#         y = self.relu(self.bn2(self.t_conv3(y)))
#         y = self.relu(self.bn1(self.t_conv2(y)))
#         y = self.t_conv1(y)
#
#         return y, x


# def test():
#     batch_size = 64
#     img_channels = 1
#     img_H, img_W = 501, 501
#     net = autoencoderNet()
#     x = torch.randn(batch_size, img_channels, img_H, img_W)
#     y, x = net(x)
#     y.to('cuda')
#     x.to('cuda')
#     print(y.shape)
#     print(x.shape)
#
# test()

# =========================
# training for auto encoder
# =========================

def train(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
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

            # forward pass: compute the output (reconsruction)
            recon = net(inputs)[0]

            # calculate the batch loss
            loss = reconLoss(recon, inputs)

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
                inputs = Variable(data[0]).cuda()

                # forward pass: compute the output (reconsruction)
                recon = net(inputs)[0]

                # calculate the batch loss
                loss = reconLoss(recon, inputs)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1

            if dir_tb is not None:
                img_grid = torchvision.utils.make_grid(torch.cat((inputs[:8], recon[:8])))
                writer.add_image('input v.s. output', img_grid)

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 10)

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



