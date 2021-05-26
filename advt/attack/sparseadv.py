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
Attack type:
Target type:

Paper:
URL:
"""
from advt.model.cnn_rnn import *
from advt.attack.attack import Attack
from advt.utils import l21_batch_loss

DEFAULT_MASK = [3, 4, 5, 11, 12, 13, 17, 18, 19]

class SparseAdv(Attack):
    def __init__(self, model, device, max_iter=100, mask=DEFAULT_MASK):
        """
        Initialize the SparseAdv class.

        Args:
            model: torch model
            device: torch.device
            max_iter: int, iteration number
            mask: list[int], the frames to be selectd
        """
        super(SparseAdv, self).__init__(model, device)
        self.max_iter = max_iter
        self.mask = mask
        self.distance_type = 'l21'

    def attack(self, xs:torch.tensor, lbl:torch.tensor):
        """
        Attack the input video-tensor by using sparse adv method.

        Args:
            xs: torch.tensor, [batch_size, frames, channel, width, height]
            lbl: torch.tensor, [batch_size, scalar]

        Returns:
            adv_xs: torch.tensor, [batch_size, frames, channel, width, height]

        """
        # move tensor to device
        origin_X = xs.detach().clone().to(self.device)
        xs = xs.detach().clone().to(self.device)
        lbl = lbl.squeeze().to(self.device)  # squeeze for crossentropyloss

        # get initializing noise for single attack
        batch_num, frame_num, channel_num, img_x, img_y = xs.shape
        init_noise = torch.zeros((batch_num, frame_num, 3, img_x, img_y)).to(self.device)
        for i in range(frame_num):
            init_noise[:, i, :, :, :] = 0.00001

        xs.requires_grad = True

        # optimizer setting
        optimizer = torch.optim.Adam([xs], lr=0.001)

        for i in range(self.max_iter):
            # # add noise to X for initializing
            if i == 0:
                xs.data = xs.data + init_noise.data

            # calculate l2,1-norm loss
            l21_loss = l21_batch_loss(xs, origin_X)

            # model forward process
            output = self.model_forward(xs)

            # crossentropy loss calculating
            softmax_pred = F.softmax(output[0])
            true_pred = softmax_pred[lbl.item()]
            loss_attack = - torch.log(1 - true_pred)

            # calculate total loss
            loss = loss_attack + l21_loss * 0.0001

            loss.backward()
            # remove invalid gradients
            gradient = xs.grad.data
            gradient[gradient != gradient] = 0
            optimizer.step()

            # model forward by using adversarial samples
            output = self.model_forward(xs)
            _, y_pred_adv = output.max(1)

            if y_pred_adv != lbl:
                print(i, 'kicked! attack success!')
                del output, y_pred_adv, gradient, _
                torch.cuda.empty_cache()
                break
            if i == self.max_iter - 1:
                print('attack failed')

            # del output, y_pred_adv, gradient, Xcnn, _
            del output, y_pred_adv, _
            torch.cuda.empty_cache()

        self.config_distance(origin_X, xs)

        return xs