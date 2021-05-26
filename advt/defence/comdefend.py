# MIT License
#
# Copyright (c) [2021] [WindFantasy98]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements Comdefend.
Defense Type: image denoising.

Paper:
URL:
"""
import os
import torch
import torch.nn as nn
from advt.defence.defence import Defend

class conv(nn.Module):
    """
    Basic conv block.
    """
    def __init__(self, inc, outc, flag=True):
        """
        Initialize the conv class.

        Args:
            inc: int, in channels
            outc: int, out channels
            flag: bool, use av_layer or not
        """
        super(conv, self).__init__()
        self.flag = flag
        self.conv = nn.Conv2d(inc, outc, kernel_size=3, bias=True, stride=1, padding=1)
        self.av_layer = nn.ELU()

    def forward(self, x):
        out = self.conv(x)
        if self.flag:
            out = self.av_layer(out)
        return out

class resconv(nn.Module):
    """
    Basic residual block.

    backbone: conv2d [kernel: 3x3 stride: 1 padding: 1]
    dowmsample: conv2d [kernel: 1x1, stride: 1, padding: 1](default)
    """
    def __init__(self, inc, outc, flag=True, downsample=None):
        """
        Initialize the resconv class.

        Args:
            inc: int, in channels
            outc: int, out channels
            flag: bool, use av_layer or not
            downsample: type of downsample
        """
        super(resconv, self).__init__()
        self.flag = flag
        self.conv = nn.Conv2d(inc, outc, kernel_size=3, bias=True, stride=1, padding=1)
        self.downsample = nn.Conv2d(inc, outc, kernel_size=1, bias=False, stride=1)
        if downsample is not None:
            self.downsample = downsample
        self.av_layer = nn.ELU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv(x)
        out = out + identity
        if self.flag:
            out = self.av_layer(out)
        return out

class ComCNN(nn.Module):
    """
    ComCNN: a end-to-end network to filter input images.
    """
    def __init__(self):
        """
        Initialize the ComCNN class.
        """
        super(ComCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            conv(3, 16),
            conv(16, 32),
            conv(32, 64),
            conv(64, 128),
            conv(128, 256),
            conv(256, 128),
            conv(128, 64),
            conv(64, 32),
            conv(32, 12, flag=False)
        )
        self.av_layer = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.av_layer(out)
        return out

class RecCNN(nn.Module):
    """
    RecCNN: ComCNN with residual block.
    """
    def __init__(self):
        """
            Initialize the RecCNN class.
        """
        super(RecCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            conv(12, 32),
            conv(32, 64),
            conv(64, 128),
            conv(128, 256),
            conv(256, 128),
            conv(128, 64),
            conv(64, 32),
            conv(32, 16),
            conv(16, 3, flag=False)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        return out

class RecCNN_beta(nn.Module):
    """
        RecCNN_beta: beta version of RecCNN.
    """
    def __init__(self):
        super(RecCNN_beta, self).__init__()
        self.conv_layers = nn.Sequential(
            conv(12, 32),
            conv(32, 64),
            conv(64, 128),
            conv(128, 256),
            conv(256, 128),
            conv(128, 64),
            conv(64, 32),
            conv(32, 16),
            conv(16, 3, flag=False)
        )
        self.av_layer = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.av_layer(out)
        return out

class ResComCNN(nn.Module):
    """
        ResComCNN: ComCNN with ResNet backbone.
    """
    def __init__(self):
        super(ResComCNN, self).__init__()
        self.layers = nn.Sequential(
            conv(3, 16),
            resconv(16, 32),
            resconv(32, 64),
            resconv(64, 128),
            resconv(128, 64),
            resconv(64, 32),
            resconv(32, 12, flag=False)
        )
        self.av_layer = nn.Sigmoid()

    def forward(self, x):
        out = self.layers(x)
        out = self.av_layer(out)
        return out

class ResRecCNN(nn.Module):
    """
        ResRecCNN: RecCNN with ResNet backbone.
    """
    def __init__(self):
        super(ResRecCNN, self).__init__()
        self.layers = nn.Sequential(
            resconv(12, 32),
            resconv(32, 64),
            resconv(64, 128),
            resconv(128, 64),
            resconv(64, 32),
            resconv(32, 12),
            resconv(12, 3, flag=False)
        )
        self.av_layer = nn.Sigmoid()

    def forward(self, x):
        out = self.layers(x)
        out = self.av_layer(out)
        return out

class ComDefend(Defend):
    """
    ComDefend: ComCNN + RecCNN
    """
    def __init__(self, model, device):
        """
        Initialize the ComDefend class.

        Args:
            model: torch.model
            device: torch.device
        """
        super(ComDefend, self).__init__(model, device)
        self.comcnn = ResComCNN().to(self.device)
        self.reccnn = ResRecCNN().to(self.device)
        self.comcnn_pth_path = ''
        self.reccnn_pth_path = ''

    def load_models_parameters(self, comcnn_path, reccnn_path):
        """
        Load the trained parameters from pth file.

        Args:
            comcnn_path: str, path of comcnn model
            reccnn_path: str, path of reccnn model

        Returns:
            None

        """
        self.comcnn_pth_path = comcnn_path
        self.reccnn_pth_path = reccnn_path

        if os.path.exists(self.comcnn_pth_path):
            self.comcnn.load_state_dict(torch.load(self.comcnn_pth_path))
            print('comcnn parameters loaded!')
        else:
            raise FileExistsError('comcnn .pth file not exist!')

        if os.path.exists(self.reccnn_pth_path):
            self.reccnn.load_state_dict(torch.load(self.reccnn_pth_path))
            print('reccnn parameters loaded!')
        else:
            raise FileExistsError('reccnn .pth file not exist!')

    def train(self, train_loader, epoch=10, is_continue=False):
        """
        Train ComDefend model.

        Args:
            train_loader: torch.dataloader, data loader of train dataset.
            epoch: int, the number of iteration for training.
            is_continue: bool, if or not continue to train comdefend with existed pth file.

        Returns:
            None
        """
        if is_continue and os.path.exists(self.comcnn_pth_path):
            self.comcnn.load_state_dict(torch.load(self.comcnn_pth_path))
            print('comcnn parameters loaded!')
        if is_continue and os.path.exists(self.reccnn_pth_path):
            self.reccnn.load_state_dict(torch.load(self.reccnn_pth_path))
            print('reccnn parameters loaded!')

        optimizer1 = torch.optim.Adam(self.comcnn.parameters(), lr=0.001)
        optimizer2 = torch.optim.Adam(self.reccnn.parameters(), lr=0.001)

        mse = nn.MSELoss(reduction='none').to(self.device)

        for e in range(epoch):
            avg_loss = 0

            for imgs, lbls in train_loader:
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)

                com_out = self.comcnn(imgs)
                com_features = com_out.detach().clone().to(self.device)
                noise = torch.randn(com_features.shape).to(self.device)
                com_features = com_features - noise
                com_features = (com_features > 0.5).float()
                com_out.data = com_features.data
                rec_out = self.reccnn(com_out)

                zero_features = torch.zeros(com_out.shape).to(self.device)
                com_loss = mse(com_out, zero_features).sum().to(self.device)
                rec_loss = mse(rec_out, imgs).sum().to(self.device)

                loss = 0.00001 * com_loss + rec_loss / (2 * len(imgs))
                avg_loss += loss.item()

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()

                optimizer1.step()
                optimizer2.step()

        torch.save(self.comcnn.state_dict(), self.comcnn_pth_path)
        torch.save(self.reccnn.state_dict(), self.reccnn_pth_path)

    def defend(self, xs:torch.tensor):
        """
        Defend the attack by bit-depth-reduction method.
        The input will be torch.tensor, and it will be transformed by bit-depth-reduction first, then the input will be feed
        to the original model. This defend method will return the output.

        Args:
            xs: torch.tensor, n-dim

        Returns:
            output: torch.tensor, n-dim

        """
        xs = xs.to(self.device)

        com_out = self.comcnn(xs)
        com_features = com_out.detach().clone()
        noise = torch.randn(com_features.shape).to(self.device)
        com_features = com_features - noise
        com_features = (com_features > 0.5).float()
        com_out.data = com_features.data
        output = self.reccnn(com_out)

        return output

if __name__ == '__main__':
    net = ResComCNN()
    print(net)
