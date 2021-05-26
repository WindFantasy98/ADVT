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
This module implements MIM.
Distance measurement includes: L-inf
Attack type: white-box
Target type: non-target

Paper:
URL:
"""

import torch
from torch.nn import CrossEntropyLoss
from advt.attack.attack import Attack
from advt.utils import l2_norm

class MIM(Attack):
    def __init__(self, model, device, decay_factor=1.0, eps=0.0001, itr_numbers=20):
        """
        Initializing the BIM class.

        Args:
            model: torch model
            device: torch.device
            decay_factor: float, momentum parameter
            eps: float, epsilon
            itr_numbers: int, iteration number
        """
        super(MIM, self).__init__(model, device)
        self.device = device
        self.eps = eps
        self.itr_numbers = itr_numbers
        self.decay_factor = decay_factor
        self.loss = CrossEntropyLoss()

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
        ys = ys.to(self.device)

        perturbtion = torch.zeros(xs.shape).to(self.device)
        xs.requires_grad = True

        for i in range(self.itr_numbers):
            output = self.model_forward(xs + perturbtion)

            _, indices = output.max(1)
            if indices != ys:
                break

            self.model_zero_grad()
            loss_val = self.loss(output, ys)
            loss_val.backward()
            grad = xs.grad.data
            x = self.eps / (l2_norm(grad) * self.itr_numbers)
            step_perturbtion = self.decay_factor * perturbtion + self.eps / (l2_norm(grad) * self.itr_numbers) * grad.sign()    # update total perturbation
            perturbtion += step_perturbtion

        adv_xs = xs + perturbtion
        adv_xs.clamp_(0, 1)

        return adv_xs
