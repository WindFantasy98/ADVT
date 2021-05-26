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
This module define a basic attack class.
"""
import os
import torch
import torch.nn as nn
from advt.utils import l0_distance, l1_distance, l2_distance, linf_distance, l21_batch_loss

class Attack():
    """
    base class for attack method.
    """
    def __init__(self, model, device):
        """
        Initializing the basic attack class.

        Args:
            model: Victim model.
            device: Device used by tensors.

        """
        if not isinstance(model, list):
            self.model = model
            self.models = None
        else:
            self.model = None
            self.models = model
        self.device = device
        self.total_distance = 0     # total perturbation distance
        self.total_num = 0          # total perturbation number
        self.distance_type = 'l2'   # distance metric type
        self.attack_parameters = {} # method parameters
        self.dis_func = {
            'l0':l0_distance,
            'l1':l1_distance,
            'l2':l2_distance,
            'linf':linf_distance,
            'l21':l21_batch_loss,
        }

    def model_zero_grad(self):
        """
        zero_grad for victim model.

        Returns:
            None

        """
        if self.models is None:
            self.model.zero_grad()
        else:
            self.seq_models_zero_grad()

    def seq_models_zero_grad(self):
        """
        clear all grads for seq models.

        Returns:
            None

        """
        for model in self.models:
            model.zero_grad()

    def model_forward(self, xs:torch.tensor):
        """
        forward process of victim model.

        Args:
            xs: input samples. n-dim torch tensor, like [batch, channel, width, height]

        Returns:
            out: adversarial samples. n-dim torch tensor, like [batch, channel, width, height]

        """
        if self.models is not None:
            return self.seq_models_forward(xs)
        return self.model(xs)

    def seq_models_forward(self, xs:torch.tensor):
        """
        if the victim model is a sequence of models, inputs will be put through this method.
        the flow of a sequence models will be like: model1 -> model2 -> model3 -> ...

        Args:
            xs: input samples. n-dim torch tensor, like [batch, channel, width, height]

        Returns:
            out: adversarial samples. n-dim torch tensor, like [batch, channel, width, height]

        """
        num_models = len(self.models)

        out = self.models[0](xs)
        for i in range(1, num_models):
            out = self.models[i](out)

        return out

    def config_distance_type(self, dis_type):
        """
        Config the metric for distance.

        Args:
            dis_type: str, 'l0', 'l1', 'l2', 'linf', 'l21'

        Returns:

        """
        self.distance_type = dis_type

    def config_distance(self, im_0, im_1):
        """
        Config distance between input and output during training.

        Args:
            im_0: torch.tensor, input
            im_1: torch.tensor, output

        Returns:

        """
        batch_num = len(im_0)
        self.total_num += batch_num

        dis_func = self.dis_func[self.distance_type]
        if self.distance_type != 'linf':
            self.total_distance += dis_func(im_0, im_1)
        else:
            self.total_distance = max(self.total_distance, dis_func(im_0, im_1))

    def display(self, tensor_im):
        """
        Display the tensor_im.

        Args:
            tensor_im: torch.tensor, 3-dim image input, [c, w, h]

        Returns:
            None
        """
        from torchvision.transforms import ToPILImage
        topil = ToPILImage()
        im = topil(tensor_im)
        im.show()

    def save_to_path(self, output_dir, id, tensor_ims, true_lbls):
        """
        Save tensor-like images / lbls to specific path.

        Args:
            output_dir: str, directory path to save
            id: int, id of the first img in a batch
            tensor_im: torch.tensor, 4-dim, [batch, c, w, h]
            tensor_lbl: torch.tensor, 2-dim, [batch, 1]

        Returns:

        """
        lbl_lst = [str(x.item()) for x in true_lbls]

        for lbl in lbl_lst:
            lbl_dir = os.path.join(output_dir, lbl)
            if not os.path.exists(lbl_dir):
                os.mkdir(lbl_dir)

        for i in range(len(tensor_ims)):
            from torchvision.transforms import ToPILImage
            topil = ToPILImage()

            im = topil(tensor_ims[i])
            im_path = os.path.join(output_dir, str(true_lbls[i].item()), str(id+i)+'.jpg')
            im.save(im_path)
