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
This module implements fgsm.
Distance measurement includes:
Attack type: white-box
Target type: non-target

Paper:
URL:
"""

import torch
import torch.nn as nn
from advt.attack.attack import Attack

class FGSM(Attack):
    def __init__(self, model, device, eps=0.001, flag_target=False):
        """
        Initializing the FGSM class.

        Args:
            model: torch model
            device: torch.device
            eps: float, epsilon
            flag_target: bool, target flag
        """
        super(FGSM, self).__init__(model, device)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.eps = eps
        self.flag_target = flag_target

    def attack(self, xs:torch.tensor, ys:torch.tensor):
        """
        Attacking the victim model by adding adversarial perturbation to test samples.
        This attack method will calculate perturbation , and return adversarial samples.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]
            ys: correct labels for input.

        Returns:
            adv_xs: adversarial samples. 4-dim torch tensor, like [batch, channel, width, height]

        """
        xs = xs.to(self.device)
        xs.requires_grad = True

        # network forward
        output = self.model_forward(xs)
        self.model_zero_grad()
        loss_val = self.loss(output, ys)
        loss_val.backward()

        # fast gradient desent method
        xs_grad = xs.grad.data
        pertubed_xs = xs + xs_grad.sign() * self.eps
        pertubed_xs.clamp_(min=0, max=1)

        # distance record
        self.config_distance(xs, pertubed_xs)

        return pertubed_xs

