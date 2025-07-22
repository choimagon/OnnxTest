from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import os

# 변환기 정의
to_pil = transforms.ToPILImage()

# 저장 함수
def save_mnist_images(split='train'):
    dataset = MNIST(root='data', train=(split=='train'), download=True)
    base_path = os.path.expanduser(f'~/mnist_png/{split}')
    for idx, (img, label) in enumerate(dataset):
        label_dir = os.path.join(base_path, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img.save(os.path.join(label_dir, f"{idx}.png"))

save_mnist_images('train')
save_mnist_images('test')
