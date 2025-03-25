import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Apple Silicon에서 MPS(Metal Performance Shaders) 사용 가능 여부 확인
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self): # 신경망 계층 초기화
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x): # 입력 데이터에 대한 연산 구현
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device) # GPU로 옮김
print(model) # 구조 출력

X = torch.rand(1, 28, 28, device=device) # 0~1 사이의 랜덤 값을 가지는 1개의 샘플(28x28) 텐서를 생성
logits = model(X) # 모델을 통해 logit(출력값)으르 계산
pred_probab = nn.Softmax(dim=1)(logits) # Softmax를 사용해 확률을 변환
y_pred = pred_probab.argmax(1) # 가장 높은 확률을 가진 클래스를 선택 및 출력
print(f"Predicted class: {y_pred}")