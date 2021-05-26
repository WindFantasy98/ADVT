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
This module implements .
Distance measurement includes:
Attack type: white-box
Target type: non-target

Paper:
URL:
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from advt.attack.attack import Attack

class DeepFool(Attack):
    def __init__(self, model, device, max_iter=50):
        """
        Initialize the DeepFool class.

        Args:
            model: torch model
            device: torch.device
            max_iter: int, the number of iteration
        """
        super(DeepFool, self).__init__(model, device)
        self.max_iter = max_iter
        self.loss = CrossEntropyLoss()

    def attack(self, xs: torch.tensor, ys: torch.tensor):
        """
        Attacking the victim model by adding adversarial perturbation to test samples.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]
            ys: 2-dim tensor, like [batch, 1], correct labels for input.

        Returns:
            xs: adversarial samples.
        """
        # initialize inputs
        xs = xs.detach().clone().to(self.device)
        ys = ys.detach().clone().to(self.device)
        orign_xs = xs.detach().clone().to(self.device)
        xs.requires_grad = True

        # perturbation = torch.zeros(xs.shape).to(self.device)
        mask = torch.ones(ys.shape).to(self.device)

        for i in range(self.max_iter):
            output = self.model_forward(xs)
            # output = self.model(xs + perturbation)
            # w = self.l1_norm(xs + perturbation)
            _, indice = output.max(1)
            if indice != ys:
                break

            xs_loss = self.loss(output, ys)
            self.model_zero_grad()
            xs_loss.backward()

            grad = xs.grad.data
            w = self.l1_norm(xs)
            xs.data = xs.data + self.f(output, ys) / w * grad.sign()

        self.config_distance(orign_xs, xs)

        return xs

    def l1_norm(self, w):
        """
        Calculate l1-norm of input matrix.

        Args:
            w: n-dim matrix.

        Returns:
            out: scalar
        """
        return torch.abs(w).sum()

    def f(self, outputs, labels):
        """
        f fuction mentioned in paper. more percisely , f6.

        Args:
            outputs: logit-layer's output. 2-dim tensor.
            labels: input's correct label. 2-dim tensor.

        Returns:
            out: scalar.
        """
        # get correct labels' one-hot vector
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        # get max value of uncorrect labels' one-hot vector
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        # get correct label's confidence
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.abs(j-i)
