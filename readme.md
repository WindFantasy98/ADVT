# Attack & Defence on Video Tasks

## Overview
**ADVT** (Attack Defence on Video Tasks) is an adversarial attack/defence toolbox open source library based on Pytorch.
This repository mainly implements some adversarial attack & defence algorithms and provides some video processing apis.

![image-20210526105234292](https://github.com/WindFantasy98/ADVT/blob/main/docs/images/image-20210526105234292.png)


## Features
The ADVT library has five functional features, which cover the whole process:  
- Preprocess  
- Attack  
- Defence  
- Record  
- Visualization  

## Attack
This module implements attack methods. All attack methods are from top computer conferences in recent 5 years:  
- [x] FGSM [Explaining and harnessing adversarial examples]
- [x] BIM []
- [x] MIM [Boosting adversarial attacks with momentum(CVPR-18)]
- [x] DeepFool [A simple and accurate method to fool deep neural networks]
- [x] DIM [Countering adversarial images using input transformations]
- [x] C&W [Towards evaluating the robustness of neural networks]
- [x] Universal [Universal adversarial perturbations(CVPR)]
- [ ] ZOO [ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models]
- [x] Sparse ADV [Sparse Adversarial Perturbations for Videos(AAAI-19)]

## Defence
This module implements defence methods.
- [x] Bit-depth Reduction
- [x] Total Variance Minimization
- [ ] Image Quilting  
- [x] ComDefend [ComDefend: An Efficient Image Compression Model to Defend Adversarial Examples(CVPR-18)]
- [x] Randomization [Mitigating Adversarial Effects Through Randomization(ICLR-18)]

## Evaluation & Visualization
ADVT provides some useful evaluation & visualization tools.
**Evaluation Metric:**
- [ ] PSNR
- [ ] SSMI
- [ ] mAP

**Visualization:**
- [x] Video-to-Frames
- [x] Frames-to-Video
- [x] Changed-Pixel

## Installation
Before installation, make sure you install fellow requirements.  
```shell script
numpy==1.18.5
opencv-python==4.4.0.42
torch==1.7.0
urllib==1.26.4
```
You can install advt through pypi or build from source code.  
**1. from pypi**
```shell script
pip install advt
```
**2. from source code**
```shell script
git clone https://github.com/WindFantasy98/ADVT.git
pip install -e .
```

## Usage
1. attack example
```python
# python3.7 torch1.7
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from advt.model.cnn import CNN
from advt.attack import FGSM

PATH_PARAMETERS = 'tests/cnn_model.pth'

def main():
    transform = transforms.Compose([transforms.ToTensor()])
    t = transforms.Compose([transforms.ToPILImage()])
    test_dataset = datasets.CIFAR10(root='/data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CNN()
    net.load_state_dict(torch.load(PATH_PARAMETERS))
    net = net.to(device)

    fgsm = FGSM(net, device)

    attack_succ = 0
    total_num = 0

    for i, (img, lbl) in enumerate(test_loader):
        img, lbl = img.to(device), lbl.to(device)
        adv_img = fgsm.attack(img, lbl)

        output = net(adv_img)
        _, pred_indice = output.max(1)

        total_num += len(lbl)
        attack_succ += (pred_indice == lbl).sum().item()
        if (i + 1) % 20 == 0:
            print('batch {}:'.format((i + 1) // 20),
                  'total tested number: {}, correct number: {}'.format(total_num, attack_succ))

if __name__ == '__main__':
    main()
```

2. defence example
```python
# python3.7 torch1.7
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from advt.model.cnn import CNN
from advt.attack import DeepFool
from advt.defence import Randomization

PATH_PARAMETERS = 'tests/cnn_model.pth'

def main():
    # initialize dataset
    transform = transforms.Compose([transforms.ToTensor()])
    t = transforms.Compose([transforms.ToPILImage()])
    test_dataset = datasets.CIFAR10(root='/data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load victim model
    net = CNN()
    net.load_state_dict(torch.load(PATH_PARAMETERS))
    net = net.to(device)

    # initialize attack method
    df = DeepFool(net, device)
    # initialize defend method
    rand_defend = Randomization(net, device)

    # initialize indicator
    attack_succ = 0
    total_num = 0

    # start attack
    for i, (img, lbl) in enumerate(test_loader):
        img, lbl = img.to(device), lbl.to(device)
        adv_img = df.attack(img, lbl)  # get adv sample

        output = rand_defend.defend(adv_img)  # get processed sample
        _, pred_indice = output.max(1)

        total_num += len(lbl)
        attack_succ += (pred_indice == lbl).sum().item()
        if (i + 1) % 20 == 0:
            print('batch {}:'.format((i + 1) // 20),
                  'total tested number: {}, correct number: {}'.format(total_num, attack_succ))

if __name__ == '__main__':
    main()
```

## Acknowledgement  
This repo is still under maintenance. For more information, please contact with me.
