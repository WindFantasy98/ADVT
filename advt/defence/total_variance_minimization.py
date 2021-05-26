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

Paper:
URL:
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
from advt.defence.defence import Defend

class TV(Defend):
    """
    TV: Total Variance
    """
    def __init__(self, model, device, test_dir, niters=30):
        """
        Initialize the TV class.

        Args:
            model: torch model
            device: torch.device
            test_dir: directory to be handled
            niters: int, iteration number
        """
        self.model = model
        self.device = device
        self.niters = niters
        self.test_dir = test_dir

    def defend(self):
        """
        Apply tv algorithm to the imgs of all labels in test_dir.

        """
        for file in os.listdir(self.test_dir):
            lbl_dir = os.path.join(self.test_dir, file)
            self.denoise_dir_imgs(lbl_dir)

    def denoise_dir_imgs(self, path):
        """
        Denoise all images in specific path.

        Args:
            path: str, directory of images

        Returns:
            None

        """
        for f in os.listdir(path):
            f_path = os.path.join(path, f)
            im = cv2.imread(f_path)
            im = im / 255
            de_im = self._tv(im) * 255

            # warning: image write back
            de_im = de_im.astype(np.uint8)
            de_im = Image.fromarray(de_im)
            de_im.save(f_path)

    def _tv(self, image, weight=0.1, eps=2.e-4, n_iter_max=200):
        """
        Perform total variance minimization on a n-dim image.

        Args:
            image: ndarray, input image, shape [h, w, c], pixel value range [0, 1]
            weight: float
            eps: float
            n_iter_max: int

        Returns:
            out: ndarray, output image, shape [h, w, c], pixel value range [0, 1]

        """
        if image.dtype == np.uint8:
            image = image / 255
        ndim = image.ndim
        p = np.zeros((image.ndim,) + image.shape, dtype=image.dtype)
        g = np.zeros_like(p)
        d = np.zeros_like(image)
        i = 0
        while i < n_iter_max:
            if i > 0:
                # d will be the (negative) divergence of p
                d = -p.sum(0)
                slices_d = [slice(None), ] * ndim
                slices_p = [slice(None), ] * (ndim + 1)
                for ax in range(ndim):
                    slices_d[ax] = slice(1, None)
                    slices_p[ax + 1] = slice(0, -1)
                    slices_p[0] = ax
                    d[tuple(slices_d)] += p[tuple(slices_p)]
                    slices_d[ax] = slice(None)
                    slices_p[ax + 1] = slice(None)
                out = image + d
            else:
                out = image
            E = (d ** 2).sum()

            # g stores the gradients of out along each axis
            # e.g. g[0] is the first order finite difference along axis 0
            slices_g = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_g[ax + 1] = slice(0, -1)
                slices_g[0] = ax
                g[tuple(slices_g)] = np.diff(out, axis=ax)
                slices_g[ax + 1] = slice(None)

            norm = np.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...]
            E += weight * norm.sum()
            tau = 1. / (2. * ndim)
            norm *= tau / weight
            norm += 1.
            p -= tau * g
            p /= norm
            E /= float(image.size)
            if i == 0:
                E_init = E
                E_previous = E
            else:
                if np.abs(E_previous - E) < eps * E_init:
                    break
                else:
                    E_previous = E
            i += 1
        return out
