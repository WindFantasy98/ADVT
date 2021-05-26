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
This module implements tools for metric evaluation/result visualization.
Result visualization methods include:
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

def figs_contrast_display(figs, titles, suptitle=''):
    """
    Show figs in one line.

    Args:
        figs: list, numpy array [H, W, C], figs to display
        titles: list, str, titles to display
        suptitle: str

    Returns:
        plt: pyplot object

    """
    if len(figs) != len(titles):
        raise ValueError('length of figs not equal length of titles')

    length = len(figs)
    plt.figure(figsize=(1, length))

    # subplot config
    for i in range(1, length+1):
        plt.subplot(1, length, i)
        plt.title(titles[i-1])
        plt.imshow(figs[i-1])

    return plt

def pixels_diff(img1, img2, is_display=False):
    """
    Calculate the difference of two images' total channels. If needed, display them in one line.

    Args:
        src_img: numpy array [H, W, C]
        hdl_img: numpy array [H, W, C]
        is_display: bool, default: False

    Returns:
        diff:  numpy array [H, W, C], src_img - hdl_img

    """
    diff = np.abs(img1.astype(np.int) - img2.astype(np.int))
    diff = diff.astype(np.uint8)

    if is_display:
        f = figs_contrast_display([img1, img2, diff], ['before', 'after', 'diff'])
        f.show()

    return diff

def pixels_chg(src_img, hdl_img, is_display=False, display_color='white'):
    """
    Display the changed pixel between src_img and hdl_img.

    Args:
        img1: numpy array [H, W, C]
        img2: numpy array [H, W, C]
        is_display: bool, default: False
        display_color: str, color of changed pixel, default:'white'

    Returns:
        diff_: numpy array [H, W, C]

    """
    mask = (src_img - hdl_img) == 0    # get changed mask
    chg = src_img * mask  # mask the src img

    # get changed pixel image matrix with specific color
    color_arr = {
        'white' : [255, 255, 255],
        'black': [0, 0, 0],
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255]
    }
    # fill the masked-img with specific color
    color_mat = np.zeros(src_img.shape).astype(np.uint8)
    pixel_color = color_arr[display_color]
    for i in range(3):
        color_mat[:, :, i] = pixel_color[i]

    chg = chg + (1 - mask) * color_mat    # add masked color matrix to the original image

    if is_display:
        f = figs_contrast_display([src_img, hdl_img, chg], ['img1', 'img2', 'changed pixel'])
        f.show()

    return chg


if __name__ == '__main__':

    from advt.utils.noise import add_ps_noise
    from advt.defence.total_variance_minimization import TV

    # img = cv2.imread('../lena.jpg')
    # img1 = add_ps_noise(img, prob=0.002)
    #
    # # f = figs_contrast_display([img, img1, img], ['a', 'b', 'a'])
    # # f = figs_contrast_show([img], ['a'])
    # # f.show()
    #
    # # d = pixels_diff(img, img1, is_display=True)
    # d = pixels_chg(img, img1, is_display=True, display_color='green')

    # test 2
    t = TV('../testdir')
    img = cv2.imread('../lena.jpg')
    img1 = add_ps_noise(img)
    img2 = t._tv(img)
    f = figs_contrast_display([img, img1, img2], ['before', 'noise', 'denoise'])
    f.show()
