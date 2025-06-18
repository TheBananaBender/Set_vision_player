import torch
from PIL import Image
from torchvision.transforms import v2
import random as rand
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

COLOR = {'Red': 0, 'Green': 1, 'Purple': 2}
SHAPE = {'Diamond': 0, 'Squiggle': 1, 'Oval': 2}
NUMBER = {'One': 0, 'Two': 1, 'Three': 2}
SHADING = {'Solid': 0, 'Striped': 1, 'Open': 2}

def img_augment(image):
    # Convert to tensor and normalize
    to_tensor = v2.Compose([
        v2.Resize((256, 256), antialias=True),
        v2.ToImage(),  # PIL -> CHW float tensor
        v2.ToDtype(torch.float32, scale=True)
    ])
    img_tensor = to_tensor(image)

    mean = [rand.uniform(0.4, 0.6) for _ in range(3)]
    std = [rand.uniform(0.2, 0.3) for _ in range(3)]
    # --- Safe augmentations ---
    augment = v2.Compose([
        v2.RandomAffine(degrees=rand.randint(0, 90), translate=(0.05, 0.05), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=rand.uniform(0.2, 0.8), contrast=rand.uniform(0.2, 0.8), saturation=rand.uniform(0.2, 0.8), hue=rand.uniform(0.01, 0.1)),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean,
                     std),
    ])

    return img_tensor,augment(img_tensor)

class SetCardDataset(Dataset):
    def __init__(self, root_folder):
        self.paths = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.paths) * 100  # simulate N=100 per image

    def __getitem__(self, idx):
        img_path = self.paths[idx % len(self.paths)]
        image = Image.open(img_path).convert('RGB')
        base_name = os.path.basename(img_path).split('.')[0]
        color, shape, number, shading = base_name.split('_')

        img_tensor, img_aug = img_augment(image)
        label = (
            COLOR[color],
            SHAPE[shape],
            NUMBER[number],
            SHADING[shading]
        )

        return img_aug, torch.tensor(label)

class MultiHeadEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(pretrained=True)
        self.features = base.features
        self.pool = base.avgpool
        self.flatten = nn.Flatten()
        self.dropout = base.classifier[0]
        in_features = base.classifier[1].in_features

        self.head_color = nn.Linear(in_features, 3)
        self.head_shape = nn.Linear(in_features, 3)
        self.head_number = nn.Linear(in_features, 3)
        self.head_shading = nn.Linear(in_features, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)

        return (
            self.head_color(x),
            self.head_shape(x),
            self.head_number(x),
            self.head_shading(x)
        )



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiHeadEfficientNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(SetCardDataset('cards'), batch_size=32, shuffle=True)

    for epoch in range(10):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out_color, out_shape, out_number, out_shading = model(images)

            loss = (
                    criterion(out_color, labels[:, 0]) +
                    criterion(out_shape, labels[:, 1]) +
                    criterion(out_number, labels[:, 2]) +
                    criterion(out_shading, labels[:, 3])
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update progress bar description with current batch loss
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} completed | Total Loss = {total_loss:.4f}")
    torch.save(model.state_dict(), 'set_card_model.pth')







