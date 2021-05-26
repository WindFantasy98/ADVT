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
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def add_ps_noise(image:np.ndarray, prob=0.05):
    """
    Add pepper salt noise to the orign image.
    Noise adding strategy is as follows:
        pixel_value = 0, if rdn < prob;
        pixel_value = 255, if rdn > 1-prob;
        else: pixel_value = orign_pixel_value

    Args:
        image: input image, shape [h, w, c], pixel value range [0, 255]/[0, 1]
        prob: the noise ratio

    Returns:
        output: output image, shape [h, w, c], pixel value range [0, 255]/[0, 1]

    """
    output = np.zeros(image.shape, np.uint8)
    h, w, c = image.shape
    mask = np.random.rand(h, w)
    one_mask = mask > 1 - prob
    zero_mask = mask < prob
    orign_mask = 1 - (zero_mask + one_mask)
    orign_mask = orign_mask[:, :, np.newaxis]

    output = orign_mask * image
    if image.dtype == np.uint8:
        output[one_mask] = 255
        output[zero_mask] = 0
    else:
        output[one_mask] = 1
        output[zero_mask] = 0
    return output.astype(np.uint8)

def add_gauss_noise(image, mean=0, var=0.01):
    """
    Add gauss noise to the origin image.

    Args:
        image: input image, shape [h, w, c], pixel value range [0, 255]/[0, 1]
        mean: gauss mean
        var: gauss variance

    Returns:
        out: output image, shape [h, w, c], pixel value range [0, 255]/[0, 1]

    """
    if image.dtype == np.uint8:
        image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    if image.dtype != np.uint8:
        out = out / 255
    return out

def rgbimg_range_cvt_(img_path, mean, std):
    """
    convert an local rgb image from one value range to another.
    Warning: in place operation.

    Args:
        img_path: str, img path
        mean: float list, dim=3
        std: float list, dim=3

    Returns:
        None

    """
    im = cv2.imread(img_path).astype(np.float64)
    img = im / 255
    img = img[:, :, [2, 1, 0]]
    img = img * std
    img = img + mean
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(img_path, img)


def rgbimgs_range_cvt_(img_folder_path, mean, std):
    """
    convert an local rgb image from one value range to another.
    Warning: in place operation.

    Args:
        img_folder_path: str, folder path
        mean: float list, dim=3
        std: float list, dim=3

    Returns:
        None

    """
    for img in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, img)
        rgbimg_range_cvt_(img_path, mean, std)

if __name__ == '__main__':
    # # noise test
    # img = cv2.imread('D:\\Codes\\Git\\advt\\tests\\lena2.jpg')
    # # img = img / 255
    # img1 = add_gauss_noise(img)
    #
    # plt.figure(figsize=(1, 2))
    # plt.suptitle('add noise')
    # plt.subplot(1, 2, 1)
    # plt.title('before')
    # plt.imshow(img)
    # plt.subplot(1, 2, 2)
    # plt.title('after')
    # plt.imshow(img1)
    # plt.show()

    # rgb img test
    test_img_path = 'D:\\data\\ucf101_jpegs_256\\result_show\\000004.png'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgbimg_range_cvt_(test_img_path, mean, std)

