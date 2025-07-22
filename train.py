import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ✅ 모델 정의
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

# ✅ 데이터셋 로딩
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ✅ 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 간단한 학습 루프 (1 epoch만)
model.train()
for epoch in range(1):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
# ✅ 학습된 모델을 PyTorch .pth 파일로 저장
torch.save(model.state_dict(), "mnist_cnn.pth")  # 🔹 이 줄 추가

# ✅ 학습된 모델을 ONNX로 export
model.eval()
dummy_input = torch.randn(1, 1, 28, 28).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "mnist_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("✅ 학습된 모델 저장 완료: mnist_cnn.pth, mnist_model.onnx")
