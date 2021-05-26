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
This module implements BIM.
Distance measurement includes: L-inf
Attack type: white-box
Target type: non-target

Paper:
URL:
"""

import torch
from torch.nn import CrossEntropyLoss
from advt.attack.attack import Attack

class BIM(Attack):
    def __init__(self, model, device, eps=0.001, itr_numbers=20):
        """
        Initializing the BIM class.

        Args:
            model: victim model
            device: torch.device
            eps: float, epsilon
            itr_numbers: int, iteration number
        """
        super(BIM, self).__init__(model, device)
        self.device = device
        self.eps = eps
        self.loss = CrossEntropyLoss()
        self.itr_numbers = itr_numbers

    def attack(self, xs:torch.tensor, ys:torch.tensor):
        """
        Attacking the victim model by adding adversarial perturbation to test samples.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]
            ys: correct labels for input.

        Returns:
            adv_xs: adversarial samples. 4-dim torch tensor, like [batch, channel, width, height]
        """
        xs = xs.to(self.device)
        perturbtion = torch.zeros(xs.shape).to(self.device)
        xs.requires_grad = True

        for i in range(self.itr_numbers):
            output = self.model_forward(xs + perturbtion)
            self.model_zero_grad()
            loss_val = self.loss(output, ys)
            loss_val.backward()
            perturbtion += self.eps * xs.grad.data.sign() / self.itr_numbers    # update total perturbation

        adv_xs = xs + perturbtion
        adv_xs.clamp_(0, 1)

        self.config_distance(xs, adv_xs)

        return adv_xs
