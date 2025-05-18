import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet152_Weights
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


COLOR_CLASSES = [
    "belyi", "bezhevyi", "biryuzovyi", "bordovyi", "chernyi",
    "fioletovyi", "goluboi", "korichnevyi", "krasnyi", "oranzhevyi",
    "raznocvetnyi", "rozovyi", "serebristyi", "seryi", "sinii",
    "zelenyi", "zheltyi", "zolotoi"
]
color_to_idx = {color: idx for idx, color in enumerate(COLOR_CLASSES)}


class AvitoColorDataset(Dataset):
    def __init__(self, data, images_folder, extension, transform=None, has_target=True):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()
        self.has_target = has_target
        if not images_folder.endswith("/"):
            images_folder += "/"
        self.images_folder = images_folder
        self.extension = extension
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = str(row['id'])
        file_name = f"{sample_id}{self.extension}"
        image_path = os.path.join(self.images_folder, file_name)
        if not os.path.exists(image_path):
            alt_ext = ".jpg" if self.extension == ".png" else ".png"
            alt_path = os.path.join(self.images_folder, f"{sample_id}{alt_ext}")
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                raise FileNotFoundError(f"Не найден файл для id={sample_id}")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.has_target:
            label_str = row["target"]
            label_idx = color_to_idx[label_str]
            return image, torch.tensor(label_idx, dtype=torch.long)
        else:
            return image, sample_id, row["category"]


def get_data_loaders(train_df, val_df, test_df,
                     train_img_folder, test_img_folder,
                     train_ext=".jpg", test_ext=".png",
                     batch_size=32):
    common_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }
    train_dataset = AvitoColorDataset(data=train_df, images_folder=train_img_folder,
                                      extension=train_ext, transform=common_transforms["train"], has_target=True)
    val_dataset = AvitoColorDataset(data=val_df, images_folder=train_img_folder,
                                    extension=train_ext, transform=common_transforms["val"], has_target=True)
    test_dataset = AvitoColorDataset(data=test_df, images_folder=test_img_folder,
                                     extension=test_ext, transform=common_transforms["test"], has_target=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, 100.0 * correct / total

def predict_with_classifier(test_loader, device, model):
    results = []
    model.eval()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for images, ids, categories in tqdm(test_loader, desc="Predicting", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            for i in range(len(images)):
                sample_id = ids[i]
                sample_cat = categories[i]
                _, pred_idx = torch.max(outputs[i], dim=0)
                predicted_color = COLOR_CLASSES[pred_idx.item()]
                color_probs = {COLOR_CLASSES[c_idx]: round(float(probs[i][c_idx].item()), 3)
                               for c_idx in range(len(COLOR_CLASSES))}
                sorted_probs = sorted(color_probs.items(), key=lambda x: x[1], reverse=True)
                top3_dict = dict(sorted_probs[:3])
                results.append({
                    "id": sample_id,
                    "category": sample_cat,
                    "predict_proba": json.dumps(top3_dict, ensure_ascii=False),
                    "predict_color": predicted_color
                })
    return results


def main():
    # Пути к CSV и папкам с изображениями
    train_csv = "train_data.csv"
    test_csv = "test_data.csv"
    train_img_folder = "train_data//"
    test_img_folder = "test_data//"

    train_extension = ".jpg"
    test_extension = ".png"


    full_train_df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)
    test_df = pd.read_csv(test_csv)


    train_loader, val_loader, test_loader = get_data_loaders(
        train_df, val_df, test_df,
        train_img_folder, test_img_folder,
        train_ext=train_extension, test_ext=test_extension,
        batch_size=32
    )

    num_classes = len(COLOR_CLASSES)
    model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nЭпоха [{epoch + 1}/{num_epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"  Обучение   - Потеря: {train_loss:.4f}, Точность: {train_acc:.2f}%")
        print(f"  Валидация  - Потеря: {val_loss:.4f}, Точность: {val_acc:.2f}%")
        print("-" * 40)

    # Сохраняем информацию о весах модели в отдельный файл
    weights_file = "model_weights.txt"
    with open(weights_file, "w", encoding="utf-8") as f:
        f.write("Ключи и размеры весов модели:\n")
        for key, tensor in model.state_dict().items():
            f.write(f"{key}: {tensor.shape}\n")
    print(f"\nИнформация о весах модели сохранена в файл: {weights_file}")

    # Финальное предсказание на тестовых данных с использованием классификатора напрямую
    results = predict_with_classifier(test_loader, device, model)
    submission_df = pd.DataFrame(results, columns=["id", "category", "predict_proba", "predict_color"])
    submission_df.to_csv("submission.csv", index=False)
    print("Файл submission.csv успешно сохранён.")

if __name__ == "__main__":
    main()
