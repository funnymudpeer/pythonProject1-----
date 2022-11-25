import cv2
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from model import Model
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def laplacian(input1):
    k0 = torch.tensor([[0,-1,0], [-1,4,-1], [0,-1,0]], dtype=torch.float32).reshape(1,3,3);
    zero = torch.zeros((1, 3, 3))
    lap_kernel = torch.zeros((3, 3, 3, 3))
    lap_kernel[0, :, :, :] = torch.cat([k0, zero, zero], dim=0)
    lap_kernel[1, :, :, :] = torch.cat([zero, k0, zero], dim=0)
    lap_kernel[2, :, :, :] = torch.cat([zero, zero, k0], dim=0)
    output1 = nn.functional.conv2d(input1, lap_kernel, padding='same')
    return output1

# 损失函数
def loss_fn(input1, input2):
    k = 0.5;
    l1_loss = torch.mean((input1-input2)**2)
    input1_gradient = laplacian(input1)
    gradient_constraint = torch.mean(input1_gradient**2)
    return l1_loss + k * gradient_constraint


if __name__ == '__main__':
    trans = transforms.ToTensor()
    blurred_cv2 = cv2.imread('./blurred1.png')
    blurred = trans(blurred_cv2)
    model = Model()
    optimizer = AdamW(model.parameters(), lr=0.05)
    # optimizer  = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-2)
    sheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=False, threshold=0.0001, patience=5)
    all_epoch = 200
    min_loss = 66666
    early_stop = 0;
    for current_epoch in range(all_epoch):
        model.train()
        optimizer.zero_grad()
        myblur = model()
        loss = loss_fn(myblur, blurred)
        loss.backward()
        optimizer.step()
        sheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch:{current_epoch} loss: {loss:.6f},  lr: {lr}')
        if current_epoch % 10 == 1:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
        if loss < min_loss:
            min_loss = loss
            best_result = model.ori;
        else:
            early_stop += 1
            if early_stop > 10:
                break;

    final_result = best_result.detach();
    torch.save(final_result, 'tensor.pt')


