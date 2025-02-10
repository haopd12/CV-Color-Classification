import argparse
import collections
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from dataloaders.base_dataset import ColorAttribute
from models.resnet import Bottleneck, ResNetCustom
from PIL import Image
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--cp', default='/home/CV-Color-Classification/model_glass_best.pth', type=str, metavar='N',
                    help='checkpoint')
    parser.add_argument('--image', default='', type=str, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    img_path = args.image
    checkpoint = args.cp


    model = ResNetCustom(Bottleneck, [3, 4, 6, 3])
    
   
    checkpoint = torch.load(checkpoint, map_location='cuda')
    new_dict = collections.OrderedDict()
    for key in checkpoint['state_dict']:
        new_dict[key.replace('module.','')] = checkpoint['state_dict'][key]

    model.load_state_dict(new_dict)
    model.eval()
    model.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    image = Image.open(img_path).convert("RGB")
    # Lấy kích thước ảnh
    width, height = image.size
    # Cắt ảnh làm đôi theo chiều ngang
    top_half = image.crop((0, 0, width, height // 2))
    bottom_half = image.crop((0, height // 2, width, height))
    top_half = transform(top_half).unsqueeze(0)
    bottom_half = transform(bottom_half).unsqueeze(0)

    image = torch.concat([top_half, bottom_half], 0)
    image = image.cuda()
    results = model(image)
    _, pred = results.topk(1, 1, True, True)
    pred = pred.t()
    print(pred)