import cv2
import torch
from torchvision import transforms
import numpy as np
from model import Model

if __name__ == '__main__':
    trans = transforms.ToTensor()
    img = cv2.imread('./figure1.png')
    img = trans(img).reshape(1, 3, 480, 720)
    gauss = Model()
