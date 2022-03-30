import os
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from torchvision.utils import make_grid, save_image
from torchvision import datasets, transforms
from typing import List

# meriels dependencies
from DatasetGenerator import MultiDirectoryDataSequence
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Resize, Lambda, Normalize
from torch.autograd import Variable
import argparse
import pytorch_ssim
from VAEsteer import *
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('pt', help='path to trained model')
    parser.add_argument('imgdir', help='path to sample images')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    print(args.pt)
    print(args.imgdir)
    # model = Model(input_shape=(135,240), latent_dim=512)
    model = torch.load(args.pt, map_location=torch.device("cpu")).eval()
    imgs = os.listdir(args.imgdir)
    print(imgs)
    # def worker_init_fn(worker_id):
    #     np.random.seed(np.random.get_state()[1][0] + worker_id)

    transform = transforms.Compose([transforms.ToTensor()])
    # dataset = datasets.ImageFolder(args.imgdir, transform=transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    new_dir = "SSIMtests/unperturbedrun-VAEsteer4thattempt"
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    for count, i in enumerate(imgs):
        img = Image.open(f"{args.imgdir}/{i}")
        img = transform(img)[None]

        x = model(img)
        s = pytorch_ssim.SSIM()
        ssim_metric = s.forward(img, x[0]).item()
        f, axarr = plt.subplots(2, 1)
        axarr[0].imshow(torch.squeeze(x[0]).permute(1,2,0).detach())
        axarr[0].set_title(f"Image {i}, SSIM={round(ssim_metric, 2)}")
        axarr[1].imshow(torch.squeeze(x[1]).permute(1, 2, 0).detach())
        # axarr[0].set_title(f"This is axarr1")
        # plt.show()
        m = str(count) if count > 9 else "0" + str(count)
        plt.savefig(f"{new_dir}/test-{m}.jpg")
        plt.pause(.1)
        plt.close()
        # images, labels = next(iter(dataloader))
        # helper.imshow(images[0], normalize=False)


if __name__ == "__main__":
    main()