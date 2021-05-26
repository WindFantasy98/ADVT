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
This module implements a iamge denoise method: randomization.

Paper:
URL:
"""
import torch
from torchvision.transforms.functional import resize, pad
from advt.defence.defence import Defend

class Randomization(Defend):
    """
    Randomization: defend the model by apply randomization to the input.
    """
    def __init__(self, model, device, prob=0.8, crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02]):
        """
        Initialize Randomization class.

        Args:
            model: torch model, model to protect
            device: torch.device
            prob: float, the prob of randomization
            crop_lst: float list, the list of crop size
        """
        super(Randomization, self).__init__(model, device)
        self.prob = prob
        self.crop_lst = crop_lst

    def defend(self, xs:torch.tensor):
        """
        Apply randomization method to the input.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]

        Returns:
            out: output samples. 4-dim torch tensor, like [batch, channel, width, height]

        """
        xs_ = self.input_transform(xs).to(self.device)
        return self.model(xs_)

    def input_transform(self, xs:torch.tensor):
        """
        Apply the transform with the given probability.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]

        Returns:
            out: output samples. 4-dim torch tensor, like [batch, channel, width, height]

        """
        p = torch.rand(1).item()
        if p <= self.prob:
            out = self.random_resize_pad(xs)
            return out
        else:
            return xs

    def random_resize_pad(self, xs:torch.tensor):
        """
        Resize and pad input image randomly.

        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]

        Returns:
            out: output samples. 4-dim torch tensor, like [batch, channel, width, height]

        """
        rand_cur = torch.randint(low=0, high=len(self.crop_lst), size=(1,)).item()
        crop_size = 1 - self.crop_lst[rand_cur]
        pad_left = torch.randint(low=0, high=3, size=(1,)).item() / 2
        pad_top = torch.randint(low=0, high=3, size=(1,)).item() / 2

        if len(xs.shape) == 4:
            bs, c, w, h = xs.shape
        elif len(xs.shape) == 5:
            bs, fs, c, w, h = xs.shape
        w_, h_ = int(crop_size * w), int(crop_size * h)
        out = resize(xs, size=(w_, h_))

        pad_left = int(pad_left * (w - w_))
        pad_top = int(pad_top * (h - h_))
        dim = [pad_left, pad_top, w - pad_left - w_, h - pad_top - h_]
        out = pad(out, padding=dim, fill=1, padding_mode='constant')
        return out