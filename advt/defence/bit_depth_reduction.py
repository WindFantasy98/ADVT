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
This module define a basic defend class.
"""
import torch
from advt.defence.defence import Defend

class BitDepthReduction(Defend):
    """
    BitDepthReduction: Bit-Depth Reduction
    """
    def __init__(self, model, device, compressed_bit=4):
        """
        Initialize the bit-depth-reduction class.

        Args:
            model: original model
            device: torch.device
            compressed_bit: int, compressed bit

        """
        super(BitDepthReduction, self).__init__(model, device)
        self.compressed_bit = compressed_bit

    def defend(self, xs: torch.tensor):
        """
        Defend the attack by bit-depth-reduction method.
        The input will be torch.tensor, and it will be transformed by bit-depth-reduction first, then the input will be feed
        to the original model. This defend method will return the output.

        Args:
            xs: torch.tensor, n-dim

        Returns:
            output: torch.tensor, n-dim

        """
        # compress tensor
        xs_compress = self.bit_depth_reduction(xs)

        # model forward
        output = self.model(xs_compress)

        return output

    def bit_depth_reduction(self, xs: torch.tensor):
        """
        This method implements bit_depth_reduction.
        The main idea of this method is to reduce perturbation of attack by reduce image bit precises.

        Args:
            xs: torch.tensor, n-dim

        Returns:
            xs_compress: torch.tensor, n-dim

        """
        # [0, 1] to [0, 2^compressed_bits-1]
        bits = 2 ** self.compressed_bit
        xs_compress = (xs.detach() * bits).int()

        # [0, 2^compressed_bit-1] to [0, 255]
        xs_255 = (xs_compress * (255 / bits))

        xs_compress = (xs_255 / 255).to(self.device)

        return xs_compress