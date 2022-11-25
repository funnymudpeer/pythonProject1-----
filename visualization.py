import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

result = torch.squeeze(torch.load('./tensor.pt')).to('cpu').detach().numpy().transpose(1,2,0)
result = np.uint8(np.clip(result * 255, 0, 255))
cv2.imshow('result', result)
cv2.waitKey()
# result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
cv2.imwrite('result.png', result)


import cv2
from skimage.metrics import structural_similarity as ssim

def match(imfil1,imfil2):
  img1=cv2.imread(imfil1)
  (h,w)=img1.shape[:2]
  img2=cv2.imread(imfil2)
  resized=cv2.resize(img2,(w,h))
  (h1,w1)=resized.shape[:2]
  img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
  return ssim(img1,img2)

imfil1 = './result.png'
imfil2 = './figure1.png' # 不同图像请修改这里
ssim_value = match(imfil1,imfil2);
print(ssim_value)