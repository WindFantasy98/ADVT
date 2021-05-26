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
This module implements dim.
Distance measurement includes:
Attack type: white-box
Target type: non-target

Paper:
URL:
"""

import torch
from torch.nn import CrossEntropyLoss
from torchvision.transforms.functional import resize, pad
from advt.attack.attack import Attack

class DIM(Attack):
    """
    DIM: Diverse Input Method
    """
    def __init__(self, model, device, prob=1.0, eps=0.01, max_iters=50, flag_target=False,
                 crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02]):
        """
        Initialize Diverse Input Method class.

        Args:
            model: torch model
            device: torch.device
            prob: float, the probability of diverse
            eps: float, epsilon
            itr_numbers: int, the iteration number
            flag_target: bool, the target flag
            crop_lst: list[int], resize list
        """
        super(DIM, self).__init__(model, device)
        self.prob = prob
        self.eps = eps
        self.max_iters = max_iters
        self.flag_target = flag_target
        self.crop_lst = crop_lst
        self.loss = CrossEntropyLoss()

    def attack(self, xs:torch.tensor, ys:torch.tensor):
        """
        Attacking the victim model by adding adversarial perturbation to test samples.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]
            ys: input labels. 2-dim torch tensor, like [batch, channel, width, height]

        Returns:
            xs_ : output adv samples. 4-dim torch tensor, like [batch, channel, width, height]

        """
        xs = xs.to(self.device)
        ys = ys.to(self.device)
        xs_ = self.input_transform(xs)
        xs_.requires_grad = True
        origin_xs = xs.detach().clone().to(self.device)

        for i in range(self.max_iters):

            out = self.model_forward(xs_)
            _, ind = out.max(1)
            if ind != ys:
                break
            loss_ = self.loss(out, ys)

            self.model_zero_grad()
            loss_.backward()

            xs_.data = xs_.data + xs_.grad.data.sign() * self.eps / self.max_iters

        self.config_distance(xs_, origin_xs)

        return xs_

    def input_transform(self, xs:torch.tensor):
        """
        Transform the input tensor.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]

        Returns:
            xs: diversed input.

        """
        p = torch.rand(1).item()
        if p <= self.prob:
            out = self.random_resize_pad(xs)
            return out
        else:
            return xs

    def random_resize_pad(self, xs:torch.tensor):
        """
        Resize and pad the input tensor randomly.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]

        Returns:
            out: random-resize-padded samples. 4-dim torch tensor, like [batch, channel, width, height]

        """
        rand_cur = torch.randint(low=0, high=len(self.crop_lst), size=(1,)).item()
        crop_size = 1 - self.crop_lst[rand_cur]
        pad_left = torch.randint(low=0, high=3, size=(1,)).item() / 2
        pad_top = torch.randint(low=0, high=3, size=(1,)).item() / 2

        b, c, w, h = xs.shape
        w_, h_ = int(crop_size * w), int(crop_size * h)
        out = resize(xs, size=(w_, h_))

        pad_left = int(pad_left * (w - w_))
        pad_top = int(pad_top * (h - h_))
        dim = [pad_left, pad_top, w - pad_left - w_, h - pad_top - h_]
        out = pad(out, padding=dim, fill=1, padding_mode='constant')
        return out