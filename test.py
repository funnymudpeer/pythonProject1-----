import cv2
import torch
from torchvision import transforms
import numpy as np


def laplacian(input1):
    k0 = torch.tensor([[0,-1,0], [-1,4,-1], [0,-1,0]], dtype=torch.float32).reshape(1,3,3);
    zero = torch.zeros((1, 3, 3))
    lap_kernel = torch.zeros((3, 3, 3, 3))
    lap_kernel[0, :, :, :] = torch.cat([k0, zero, zero], dim=0)
    lap_kernel[1, :, :, :] = torch.cat([zero, k0, zero], dim=0)
    lap_kernel[2, :, :, :] = torch.cat([zero, zero, k0], dim=0)
    output1 = torch.nn.functional.conv2d(input1, lap_kernel, padding='same')
    return output1

if __name__ == '__main__':
    trans = transforms.ToTensor()
    img = cv2.imread('./figure1.png')
    img = trans(img).reshape(1,3,480,720)
    result = laplacian(img)
    img = torch.squeeze(result).numpy().transpose(1,2,0)
    img = np.uint8(np.clip(img*255,0,255))
    cv2.imshow('hh', img)
    cv2.waitKey()