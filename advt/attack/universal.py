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
This module implements Universal Perturbation Attack.
Distance measurement includes:
Attack type: white-box
Target type: untargeted

Paper:
URL:
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from advt.attack.attack import Attack

class Universal(Attack):
    """
    Universal: Universal Perturbation.
    """
    def __init__(self, model, device, total_num, unv_rate=0.7, max_iter=100):
        """
        Initialize the Universal class.

        Args:
            model: torch model, victim model.
            device: torch.device.
            total_num: int, total number of dataset.
            unv_rate: float, the percent of dataset need to be fooled.
        """
        super(Universal, self).__init__(model, device)
        self.unv_rate = unv_rate
        self.total_num = total_num
        self.max_iter = max_iter
        self._unv_perturbation = None

    def attack(self, xs: torch.tensor, ys: torch.tensor):
        """
        Attacking the victim model by adding adversarial perturbation to test samples.

        Args:
            xs: input tensor, input samples, 4-dim torch tensor, like [batch, channel, width, height]
            ys: 2-dim tensor, like [batch, 1], correct labels for input.

        Returns:
            None

        """
        xs = xs.to(self.device)
        ys = ys.to(self.device)

        perturbation = torch.zeros_like(xs).to(self.device)
        xs.requires_grad = True

        for i in range(self.max_iter):
            out = self.model(xs + perturbation + self._unv_perturbation)

        return xs

    @property
    def unv_perturbation(self):
        if self._unv_perturbation is None:
            raise ValueError('Perturbation is None!')
        return self._unv_perturbation

    def q(self, perturb):
        """
        f fuction mentioned in paper. more percisely , f6.

        Args:
            outputs: logit-layer's output. 2-dim tensor.
            labels: input's correct label. 2-dim tensor.

        Returns:
            out: scalar.
        """
        self.unv_perturbation = torch.clamp(self._unv_perturbation + perturb, min=0, max=1)
