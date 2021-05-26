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
This module implements many useful tools for handling video datasets.

"""
import os
import cv2
import torch
import numpy as np
from cv2 import VideoWriter_fourcc
from torchvision.transforms import ToPILImage

def video_dataset_parser(videos_dir,
                         frames_dir,
                         num_frame):
    """
    Build a dataset from existed video file path.

    Args:
        videos_dir: str, the videos' directory path, structure like:
            videos/
                -- class1/
                    -- video1.mp4
                    -- video2.mp4
                    ...
                -- class2/
                    -- video1.mp4
                    ...
                ...
        frames_dir: str, a empty directory to save extracted frames, object directory structure
                    will be like:
            dataset/
                -- class1/
                    -- video1.mp4/
                        -- frame000000.png
                        -- frame000001.png
                        ...
                    ...
                ...
        num_frame: int, the number of frames needed to extract for each video.

    Returns:
        None

    """
    assert os.path.exists(videos_dir), 'Videos path: {} not exists!'.format(videos_dir)

    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)

    # loop: traverse classes
    for video_class in os.listdir(videos_dir):
        src_class_path = os.path.join(videos_dir, video_class)
        obj_class_path = os.path.join(frames_dir, video_class)
        if not os.path.exists(obj_class_path):
            os.mkdir(obj_class_path)

        # loop: traverse videos
        for video in os.listdir(src_class_path):
            video_path = os.path.join(src_class_path, video)
            obj_cls_frame_path = os.path.join(obj_class_path, video)
            if not os.path.exists(obj_cls_frame_path):
                os.mkdir(obj_cls_frame_path)

            extract_frames(video_path, obj_cls_frame_path, num_frame)  # extract frames

def extract_frames(video_dir,
                   frame_dir,
                   num_frame,
                   size=(),
                   ):
    """
    Extract the number of frames from the specific video.

    Args:
        video_dir: str, the directory of video.
        frame_dir: str, the directory to save frames.
        num_frame: int, the number of frames need to extract.
        size: tuple, int, (width, height).

    Returns:
        None

    """
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

    vc = cv2.VideoCapture(video_dir)

    # get frames from video
    frames = []
    if vc.isOpened():
        while True:
            ret, frame = vc.read()
            if ret == False:
                break
            frames.append(frame)

    # skip number
    gap = len(frames) // num_frame
    cur = 0
    while cur < len(frames) and cur < num_frame:
        frame_name = 'frame{:06d}.png'.format(cur)
        frame_path = os.path.join(frame_dir, frame_name)
        if len(size) == 2:
            frames[cur*gap] = cv2.resize(frames[cur*gap], dsize=size)
        cv2.imwrite(frame_path, frames[cur*gap])
        cur += 1

def frames2video(frame_dir,
                 video_file,
                 fps=30,
                 fourcc='XVID',
                 filename_tmpl='{:06d}.jpg',
                 start=0,
                 end=0):
    """
    Read the frame images from a directory and join them as a video.

    Args:
        frame_dir (str): The directory containing video frames.
        video_file (str): Output filename.
        fps (float): FPS of the output video.
        fourcc (str): Fourcc of the output video, this should be compatible
            with the output file type.
        filename_tmpl (str): Filename template with the index as the variable.
        start (int): Starting frame index.
        end (int): Ending frame index.
        show_progress (bool): Whether to show a progress bar.
    """
    if end == 0:
        end = len(os.listdir(frame_dir))

    first_file = os.path.join(frame_dir, filename_tmpl.format(start))

    # get frame's height, width
    img = cv2.imread(first_file)
    height, width = img.shape[:2]
    resolution = (width, height)

    # config cv2's video writer
    vwriter = cv2.VideoWriter(video_file, VideoWriter_fourcc(*fourcc), fps,
                              resolution)

    # wirte frame to frame_dir
    def write_frame(file_idx):
        filename = os.path.join(frame_dir, filename_tmpl.format(file_idx))
        img = cv2.imread(filename)
        vwriter.write(img)

    for i in range(start, end):
        write_frame(i)

    # release cv2 writer
    vwriter.release()


def tensor_to_pil(tr_image: torch.tensor, is_show=False):
    """
    transfer a tensor to pil image.
    ONLY FOR CNN_RNN Normalize!

    Args:
        tr_image: torch.tensor, 3-dim
        is_show: bool, default: False
    Returns:
        image2: Image object
    """
    from PIL import Image
    img2 = tr_image.detach().cpu().numpy()
    # img2 = img2[[2, 1, 0], :, :]

    # normalize for cnn_rnn video model specially
    mean = [[[0.485]], [[0.456]], [[0.406]]]
    std = [[[0.229]], [[0.224]], [[0.225]]]
    img2 *= std
    img2 += mean

    img2 = img2.transpose(1, 2, 0)
    img2 *= 255
    img2 = img2.astype(np.uint8)
    image2 = Image.fromarray(img2)
    if is_show:
        image2.show()
    return image2

def singlevideo_save_to_path(input_tensor, output_dir):
    """
    Converts a single video tensor into pictures and save them to the specific path.

    Args:
        input_tensor: torch tensor, 5-dim tensor with shape of [1, frames, channels, height, width].
        output_dir: str valid folder dir.

    Return:
        None
    """
    index = 0
    for fig_tensor in input_tensor[0]:
        img = tensor_to_pil(fig_tensor)
        img_name = str(index).zfill(6) + '.png'
        img.save(os.path.join(output_dir, img_name))
        index += 1

def get_onehot(class_id, total_classes):
    """
    Give an integer, return a one-hot tensor

    Args:
        class_id: int, the label of class.
        total_classes: int, the total number of classes.

    Returns:
        output: torch.tensor, one-hot vector
    """
    return torch.eye(total_classes)[class_id]

    # def save_grad(tensor_name):
    #     def hook(grad):
    #         grads[tensor_name] = grad
        # return hook

if __name__ == '__main__':
    # frames2video('D:\\data\\test\\test_video\\1', 'D:\\data\\test\\test_video\\2\\test.mp4', filename_tmpl='frame{:06d}.jpg', start=1)
    extract_frames('D:\\data\\test\\test_video\\2\\test.mp4', 'D:\\data\\test\\test_video\\3', 20, size=(200, 100))
    # video_dataset_parser('D:\\data\\test\\test_dataset\\ucf101-part', 'D:\\data\\test\\test_dataset\\dataset', 20)