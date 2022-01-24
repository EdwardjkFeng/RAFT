import sys

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import gc
from torch.profiler import profile, record_function, ProfilerActivity

import time

sys.path.append('core')

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


DEVICE = 'cuda'


def load_image(imfile):
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img = np.repeat(np.array(Image.open(imfile)).astype(np.uint8)[..., np.newaxis], 3, axis=2)  # adapted for grayscale
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey(10)


def make_dir_if_not_exist(path):
    """ Make a directory if the input path does not exist
    Args:
        path (str): path to check and eventually create
    """
    if not os.path.exists(path):
        os.makedirs(path)


# ToDo: function write_flo()
def write_flo(out_file, flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    make_dir_if_not_exist(os.path.dirname(out_file))
    with open(out_file, 'wb') as object_out:
        np.array([80, 73, 69, 72], np.uint8).tofile(object_out)
        np.array([flo.shape[1], flo.shape[0]], np.int32).tofile(object_out)
        np.array(flo, np.float32).tofile(object_out)

    print('Written:', out_file)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        # images2 = sorted(glob.glob(os.path.join('/home/jingkun/SemesterProject/KITTI/data_odometry_gray/dataset/sequences/03/image_0/', '*.png')))

        images = sorted(images)
        start = time.perf_counter()
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
        # for imfile1, imfile2 in zip(images, images2):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # forward tracking
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # unpad to restore the dimensions
            flow_up = padder.unpad(flow_up)
            # viz(padder.unpad(image1), flow_up)

            # kitti
            out_file = os.path.dirname(os.path.dirname(imfile1)) + '/image' + os.path.dirname(
                imfile1)[-1] + '_flo/forward/' + "{}.{}".format(os.path.basename(imfile1)[:-4], 'flo')
            # euroc
            # out_file = os.path.dirname(os.path.dirname(os.path.dirname(imfile1))) + '/cam' + os.path.dirname(os.path.dirname(
            #     imfile1))[-1] + '_flo/forward/' + "{}.{}".format(os.path.basename(imfile1)[:-4], 'flo')
            write_flo(out_file, flow_up)

            # backward tracking
            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
            # unpad to restore the dimensions
            flow_up = padder.unpad(flow_up)
            # viz(image1, flow_up)

            # kitti
            out_file = os.path.dirname(os.path.dirname(imfile1)) + '/image' + os.path.dirname(
                imfile1)[-1] + '_flo/backward/' + "{}.{}".format(os.path.basename(imfile1)[:-4], 'flo')
            # euroc
            # out_file = os.path.dirname(os.path.dirname(os.path.dirname(imfile1))) + '/cam' + os.path.dirname(os.path.dirname(
            #     imfile1))[-1] + '_flo/backward/' + "{}.{}".format(os.path.basename(imfile1)[:-4], 'flo')
            write_flo(out_file, flow_up)

            del image1, image2, flow_low, flow_up, padder, out_file
            gc.collect()

        print("Infer optical flow pairs for {} frames in {:.4f} seconds".format(len(images), time.perf_counter()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     with record_function("model_inference"):
    #         demo(args)
    #
    # print(prof.key_averages().table(row_limit=10))
    demo(args)
