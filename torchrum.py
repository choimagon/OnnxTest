import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image
import time

# ✅ 모델 정의 (훈련 때와 동일해야 함)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ✅ 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# ✅ 전처리 정의
transform = transforms.Compose([
    transforms.Grayscale(),        # 혹시 RGB 이미지면 흑백으로
    transforms.Resize((28, 28)),
    transforms.ToTensor(),         # [0, 1], shape: (1, 28, 28)
])

# ✅ 이미지 경로 수집
root = os.path.expanduser("~/mnist_png/test/")
image_paths = []
labels = []

for digit in range(10):
    digit_dir = os.path.join(root, str(digit))
    files = sorted(os.listdir(digit_dir))
    for fname in files[:10]:  # 클래스당 10장씩
        image_paths.append(os.path.join(digit_dir, fname))
        labels.append(digit)

# ✅ 추론 및 속도 측정
correct = 0
start = time.time()

for path, label in zip(image_paths, labels):
    img = Image.open(path).convert("L")
    input_tensor = transform(img).unsqueeze(0).to(device)  # (1, 1, 28, 28)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

    if pred == label:
        correct += 1

end = time.time()
elapsed = end - start

# ✅ 결과 출력
print(f"\n✅ 총 100장 추론 완료")
print(f"정확도: {correct}/100 = {correct}%")
print(f"총 소요 시간: {elapsed:.4f}초")
print(f"이미지당 평균 시간: {elapsed * 1000 / 100:.2f} ms")
