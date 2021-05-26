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
This module define a basic defend class.
"""
import torch
import torch.nn as nn

class Defend():
    """
    Basic Defend Class
    """
    def __init__(self, model, device):
        """
        Initializing basic defend class object.

        Args:
            model: torch model
            device: torch.device
        """
        self.model = model.to(device)
        self.device = device
        self.loss_lst = {}

    def config_l2(self):
        """
        Initialize l2 loss config.

        Returns:

        """
        self.loss_lst['l2'] = []

    def config_l2_loss(self, x0:torch.tensor, x1:torch.tensor):
        """
        calculate the l2 dis of x0 tensor and x1 tensor.

        Args:
            x0: torch.tensor
            x1: torch.tensor

        Returns:
            None

        """
        l2_loss = torch.square(x0 - x1)
        self.loss_lst['l2'].append(l2_loss)
