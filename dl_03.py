import torch
import numpy as np

# Apple Silicon에서 MPS(Metal Performance Shaders) 사용 가능 여부 확인
if torch.backends.mps.is_available():
    device = torch.device("mps")  # GPU 사용
else:
    device = torch.device("cpu")  # CPU 사용
print(f"Using device: {device}")

# NumPy식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# 텐서 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 산술 연산
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
# ``tensor.T`` 는 텐서의 전치(transpose)를 반환합니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# 단일-요소(single-element) 텐서
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# 바꿔치기(in-place) 연산(권장하지 않음)
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# 텐서를 NumPy 배열로 변환
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# 텐서의 변경 사항이 NumPy 배열에 반영됨
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy 배열을 텐서로 변환
n = np.ones(5)
t = torch.from_numpy(n)

# NumPy 배열의 변경 사항이 텐서에 반영됨
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")