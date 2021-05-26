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
This module implements a iamge denoise method: total variance minimization.

"""
import torch

def l0_distance(x1:torch.tensor, x2:torch.tensor):
    return (x1 != x2).sum().item()

def l1_distance(x1:torch.tensor, x2:torch.tensor):
    return torch.abs(x1 - x2).sum().item()

def l2_distance(x1:torch.tensor, x2:torch.tensor):
    return torch.sqrt(((x1 - x2) ** 2).sum()).item()

def linf_distance(x1:torch.tensor, x2:torch.tensor):
    abs_x = torch.abs(x1 - x2)
    return abs_x.max().item()

def l1_norm(x:torch.tensor):
    """
    Calculate the l1-norm of input tensor.

    Args:
        x: torch.tensor

    Returns:
        torch.tensor, 0-dim
    """
    return torch.abs(x).sum()

def l2_norm(x:torch.tensor):
    """
    Calculate the l2-norm of input tensor.

    Args:
        x: torch.tensor

    Returns:
        out: torch.tensor, 0-dim
    """
    return torch.square(x).sum()

def l21_batch_loss(output_data, orign_data):
    """
    Calculate two tensors's l21 loss, if batch_size > 1, output should be l21 loss / batch_size
    if add mask, then input data should be: mask * tensor_data

    Args:
        output_data: [batch, frames, channels, height, width]
        orign_data: [batch, frames, channels, height, width]

    Return:
        scalar tensor, l21 loss
    """
    diff_square = (output_data - orign_data).square()
    # question: should i divide l2batchloss by batchsize ?
    l2_batch_loss = diff_square.sum(axis=(0, 2, 3, 4)).sqrt() / len(output_data)
    l21_batch_loss = l2_batch_loss.sum()
    return l21_batch_loss
