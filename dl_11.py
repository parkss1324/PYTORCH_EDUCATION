import torch
import torchvision.models as models

# 모델 생성 및 사전 학습된 가중치 다운로드
model = models.vgg16(weights='IMAGENET1K_V1')

# 모델 가중치만 저장
torch.save(model.state_dict(), 'model_weights.pth')

# 모델 구조를 새로 생성하고 가중치 로드
model = models.vgg16()  # 학습되지 않은 모델 생성
model.load_state_dict(torch.load('model_weights.pth'))  # 저장된 가중치 로드
model.eval()  # 모델을 평가 모드로 설정

# 올바른 저장 방식 (가중치만 저장)
torch.save(model.state_dict(), 'model_vgg16.pth')

# 올바른 로드 방식
model = models.vgg16()  # 다시 모델 구조 생성
model.load_state_dict(torch.load('model_vgg16.pth'))
model.eval()
