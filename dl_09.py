import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

input_image = torch.rand(3,28,28) # 28x28 크기의 이미지 3개의 미니배치
print(input_image.size())

flatten = nn.Flatten() # 2D 이미지를 연속된 배열로 변환
flat_image = flatten(input_image) # 28x28 = 784 배열로 변환
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20) # 입력 특징 수 28x28, 출력 특징 수 20
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1) # 비선형성을 도입하고, 신경망이 다양한 현상을 학습할 수 있도록 함
print(f"After ReLU: {hidden1}")