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
from advt.attack.attack import Attack

class CW(Attack):
    def __init__(self, model, device, c=1e-4, keppa=0, steps=1000, flag_target=False, is_config=True):
        """
        Initialize the CW class.

        Args:
            model: torch model
            device: torch.device
            c: float, l2 regulation parameter
            keppa: float, CW keppa
            steps: int, steps to run
            flag_target: bool,
            is_config: bool, config distance or not
        """
        super(CW, self).__init__(model ,device)
        self.device = device
        self.c = c
        self.keppa = keppa
        self.steps = steps
        self.flag_target = flag_target
        self.is_config = is_config

    def attack(self, xs:torch.tensor, ys:torch.tensor):
        """
        Attacking the victim model by adding adversarial perturbation to test samples with CW attack.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]
            ys: correct labels for input.

        Returns:
            best_adv_xs: adversarial samples. 4-dim torch tensor, like [batch, channel, width, height]

        """
        # avoid to influence original data
        xs = xs.detach().clone().to(self.device)
        ys = ys.detach().clone().to(self.device)

        # transform xs's iamge space
        w = self.inverse_tanh_space(xs).detach()
        w.requires_grad = True

        # to record best adv samples
        best_adv_xs = xs.detach().clone()
        best_l2 = 1e10 * torch.ones((len(xs))).to(self.device)
        prev_cost = 1e10
        dim = len(xs.shape)

        mseloss = nn.MSELoss(reduction='none')
        flatten = nn.Flatten()
        optimizer = torch.optim.Adam([w], lr=0.01)

        for step in range(self.steps):
            adv_xs = self.tanh_space(w)
            l2_dis = mseloss(flatten(adv_xs), flatten(xs)).sum()

            outputs = self.model_forward(adv_xs)
            _, ind = outputs.max(1)
            f_loss = self.f(outputs, ys).sum()

            cost = l2_dis + f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # update adv imgs
            _, pre = torch.max(outputs, 1)
            correct = (pre==ys).float()

            # element-wise mask to select best_adv_xs
            mask = (1 - correct) * (best_l2 > l2_dis)
            best_l2 = mask * l2_dis + (1 - mask) * best_l2

            # keep shape same
            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_xs = mask * adv_xs + (1 - mask) * best_adv_xs

            # note: only for batchsize=1!
            if ind.item() != ys.item():
                return best_adv_xs
            # each 10 steps update 1 time
            if step % (self.steps//10) == 0:
                # if pre_cost increase, then break loop
                if cost.item() > prev_cost:
                    return best_adv_xs
                prev_cost = cost.item()

        if self.is_config:
            self.config_distance(xs, best_adv_xs)

        return best_adv_xs

    def tanh_space(self, x):
        """
        transfer x from original image space to tanh space.

        Args:
            x: input image. torch tensor with value range of [0, 1]

        Returns:
            x of tanh space

        """
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        """
        transfer x from tanh space to original image space.

        Args:
            x: input tensor. torch tensor with value range of [-inf, +inf]

        Returns:
            x of original space

        """
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x)/(1 - x))

    def f(self, outputs, labels):
        """
        f fuction mentioned in paper. more percisely , f6.

        Args:
            outputs: logit-layer's output. 2-dim tensor.
            labels: input's correct label. 2-dim tensor.

        Returns:

        """
        # get correct labels' one-hot vector
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        # get max value of uncorrect labels' one-hot vector
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        # get correct label's confidence
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp((j-i), min=-self.keppa)
