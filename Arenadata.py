import os
import glob
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Используемое устройство:", device)

psx_folder = "telecom100k"
psx_csv_files = glob.glob(os.path.join(psx_folder, "*.csv"))
psx_txt_files = glob.glob(os.path.join(psx_folder, "*.txt"))
psx_files = psx_csv_files + psx_txt_files

if not psx_files:
    raise ValueError("Не найдено ни одного файла PSXStats в папке '{}'".format(psx_folder))

df_list = []
for file_path in tqdm(psx_files, desc="Чтение файлов PSXStats", dynamic_ncols=True):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, sep=",")
    elif file_path.endswith(".txt"):
        df = pd.read_csv(file_path, sep="|")
    else:
        continue
    df.columns = [col.strip() for col in df.columns]
    df_list.append(df)

df_psx = pd.concat(df_list, ignore_index=True)
print("Общее количество записей PSXStats:", len(df_psx))
print("Колонки PSXStats:", df_psx.columns.tolist())

def parse_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    for fmt in ("%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except Exception:
            continue
    return pd.NaT

df_psx["StartSession_dt"] = df_psx["StartSession"].apply(parse_date)
df_psx["Hour"] = df_psx["StartSession_dt"].dt.floor("h")

required_cols = ["IdSession", "IdSubscriber", "Hour", "Duartion", "UpTx", "DownTx"]
for col in required_cols:
    if col not in df_psx.columns:
        raise KeyError(f"Отсутствует необходимый столбец: {col}")

df_vitrine = df_psx.groupby(["Hour", "IdSubscriber"], as_index=False).agg({
    "UpTx": "sum",
    "DownTx": "sum",
    "Duartion": "sum",
    "IdSession": "count"
})
df_vitrine.rename(columns={"IdSession": "SessionCount"}, inplace=True)
df_vitrine["TotalTraffic"] = df_vitrine["UpTx"] + df_vitrine["DownTx"]

print("Пример агрегированной витрины (первые 5 строк):")
print(df_vitrine.head())

try:
    df_subscribers = pd.read_csv(os.path.join(psx_folder, "subscribers.csv"), sep=",")
    print("Колонки в subscribers.csv:", df_subscribers.columns.tolist())
    df_subscribers.rename(columns={"IdOnPSX": "IdSubscriber"}, inplace=True)
    df_subscribers["IdSubscriber"] = df_subscribers["IdSubscriber"].astype(int)
    df_vitrine = pd.merge(df_vitrine, df_subscribers, on="IdSubscriber", how="left")
except FileNotFoundError:
    print("Файл subscribers.csv не найден. Продолжаем без дополнительных данных подписчиков.")

try:
    df_ground = pd.read_csv(os.path.join(psx_folder, "RESULT"), sep=",")
    print("Файл RESULT успешно загружен. Колонки:", df_ground.columns.tolist())
    df_ground.rename(columns={"Id": "IdSubscriber"}, inplace=True)
    df_ground["IdSubscriber"] = df_ground["IdSubscriber"].astype(int)
    # Объединяем все столбцы (UID, Type, IdPlan, TurnOn, Hacked, Traffic)
    df_vitrine = pd.merge(df_vitrine, df_ground, on="IdSubscriber", how="left", suffixes=("", "_ground"))
    if "Hacked_ground" in df_vitrine.columns:
        df_vitrine["Hacked"] = df_vitrine["Hacked_ground"]
        df_vitrine.drop(columns=["Hacked_ground"], inplace=True)
    df_vitrine["Hacked"] = df_vitrine["Hacked"].fillna(False).infer_objects(copy=False)
    print("Истинные метки загружены из файла RESULT и заполнены отсутствующие как False.")
except FileNotFoundError:
    np.random.seed(42)
    df_vitrine["Hacked"] = np.random.choice([False, True], size=len(df_vitrine), p=[0.99, 0.01])
    print("Файл RESULT не найден. Метки сгенерированы случайным образом.")

df_labeled = df_vitrine[~df_vitrine["Hacked"].isna()].copy()
features = ["UpTx", "DownTx", "Duartion", "SessionCount", "TotalTraffic"]
df_labeled[features] = df_labeled[features].fillna(0)

X = df_labeled[features].values
y = df_labeled["Hacked"].astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

    def __init__(self):
            ...
            self.fc = nn.Linear(500, 100)
            self.bn = BatchNorm1d(100)
            ...

    def forward(self, x):
        ...
        x = F.relu(self.fc(x))
        x = self.bn(x)
        ...

model = SimpleNN(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with tqdm(train_loader,
              desc=f"Эпоха {epoch + 1}/{epochs}",
              leave=True,
              dynamic_ncols=True) as pbar:

        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

            preds = (outputs >= 0.5).float()
            correct = (preds == batch_y).sum().item()
            total_correct += correct
            total_samples += batch_X.size(0)

            batch_acc = correct / batch_X.size(0)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "batch_acc": f"{batch_acc:.4f}"
            })

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = total_correct / total_samples
    tqdm.write(f"[Эпоха {epoch + 1}/{epochs}] Потери: {epoch_loss:.4f}, Точность: {epoch_acc:.4f}")

model.eval()
with torch.no_grad():
    y_preds = []
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        y_preds.append(outputs.cpu())
    y_preds = torch.cat(y_preds, dim=0)
    y_pred_labels = (y_preds >= 0.5).int().numpy().flatten()
    y_true = y_test_tensor.cpu().numpy().flatten()

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_labels))
print("\nClassification Report:")
print(classification_report(y_true, y_pred_labels))

X_all_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    all_preds = model(X_all_tensor).cpu()
df_labeled["Hacked_pred"] = (all_preds >= 0.5).int().numpy()

df_final = df_labeled[df_labeled["Hacked_pred"] == 1].copy()
df_final.rename(columns={"IdSubscriber": "Id", "Hacked_pred": "Hacked"}, inplace=True)
desired_cols = ["Id", "UID", "Type", "IdPlan", "TurnOn", "Hacked", "Traffic"]
missing_cols = [col for col in desired_cols if col not in df_final.columns]
if missing_cols:
    print("В итоговом DataFrame отсутствуют столбцы:", missing_cols)
else:
    df_final = df_final[desired_cols].drop_duplicates(subset=desired_cols)
    output_filename = os.path.join(psx_folder, "FINAL_RESULT_F.csv")
    df_final.to_csv(output_filename, index=False)
    print(f"\nИтоговый файл сохранён как {output_filename}")

new_folder = "telecom1000k"
new_csv_files = glob.glob(os.path.join(new_folder, "*.csv"))
new_txt_files = glob.glob(os.path.join(new_folder, "*.txt"))
new_files = new_csv_files + new_txt_files

if not new_files:
    raise ValueError("Не найдено ни одного файла PSXStats в папке '{}'".format(new_folder))

new_df_list = []
for file_path in tqdm(new_files, desc="Чтение файлов PSXStats (telecom1000k)", dynamic_ncols=True):
    if file_path.endswith(".csv"):
        df_new = pd.read_csv(file_path, sep=",")
    elif file_path.endswith(".txt"):
        df_new = pd.read_csv(file_path, sep="|")
    else:
        continue
    df_new.columns = [col.strip() for col in df_new.columns]
    new_df_list.append(df_new)

df_new_psx = pd.concat(new_df_list, ignore_index=True)
print("Общее количество записей нового набора PSXStats:", len(df_new_psx))
print("Колонки нового набора PSXStats:", df_new_psx.columns.tolist())

df_new_psx["StartSession_dt"] = df_new_psx["StartSession"].apply(parse_date)
df_new_psx["Hour"] = df_new_psx["StartSession_dt"].dt.floor("h")

required_cols_new = ["IdSession", "IdSubscriber", "Hour", "Duartion", "UpTx", "DownTx"]
for col in required_cols_new:
    if col not in df_new_psx.columns:
        raise KeyError(f"Отсутствует необходимый столбец в новом наборе: {col}")

df_new_vitrine = df_new_psx.groupby(["Hour", "IdSubscriber"], as_index=False).agg({
    "UpTx": "sum",
    "DownTx": "sum",
    "Duartion": "sum",
    "IdSession": "count"
})
df_new_vitrine.rename(columns={"IdSession": "SessionCount"}, inplace=True)
df_new_vitrine["TotalTraffic"] = df_new_vitrine["UpTx"] + df_new_vitrine["DownTx"]

print("Пример агрегированной витрины нового набора (первые 5 строк):")
print(df_new_vitrine.head())

try:
    new_subscribers_path = os.path.join(new_folder, "subscribers.csv")
    df_new_subscribers = pd.read_csv(new_subscribers_path, sep=",")
    print("Колонки в новом subscribers.csv:", df_new_subscribers.columns.tolist())
    df_new_subscribers.rename(columns={"IdOnPSX": "IdSubscriber"}, inplace=True)
    df_new_subscribers["IdSubscriber"] = df_new_subscribers["IdSubscriber"].astype(int)
    if "IdClient" in df_new_subscribers.columns:
        df_new_subscribers.rename(columns={"IdClient": "UID"}, inplace=True)
    else:
        print("В новом subscribers.csv отсутствует столбец idclient для UID.")
    df_new_vitrine = pd.merge(df_new_vitrine, df_new_subscribers, on="IdSubscriber", how="left")
except FileNotFoundError:
    print("Файл subscribers.csv в новом наборе не найден. Продолжаем без дополнительных данных подписчиков.")

try:
    company_path = os.path.join(new_folder, "company.parquet")
    physical_path = os.path.join(new_folder, "physical.parquet")
    df_company = pd.read_parquet(company_path)
    df_physical = pd.read_parquet(physical_path)
    if "Type" not in df_company.columns:
        df_company["Type"] = "J"
    if "Type" not in df_physical.columns:
        df_physical["Type"] = "P"
    df_company.rename(columns={"Id": "UID"}, inplace=True)
    df_physical.rename(columns={"Id": "UID"}, inplace=True)
    df_company = df_company[["UID", "IdPlan", "Type"]]
    df_physical = df_physical[["UID", "IdPlan", "Type"]]
    df_info = pd.concat([df_company, df_physical], ignore_index=True)
    df_new_vitrine = pd.merge(df_new_vitrine, df_info, on="UID", how="left", suffixes=("", "_info"))
except Exception as e:
    print("Ошибка при загрузке или объединении с company.parquet/physical.parquet:", e)

df_new_vitrine["TurnOn"] = True

traffic_series = df_new_vitrine.groupby("UID")["TotalTraffic"].sum().rename("Traffic_total")
df_new_vitrine = pd.merge(df_new_vitrine, traffic_series, on="UID", how="left")
df_new_vitrine["Traffic"] = df_new_vitrine["Traffic_total"]
df_new_vitrine.drop(columns=["Traffic_total"], inplace=True)

new_features = ["UpTx", "DownTx", "Duartion", "SessionCount", "TotalTraffic"]
df_new_vitrine[new_features] = df_new_vitrine[new_features].fillna(0)
X_new = df_new_vitrine[new_features].values
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    new_preds = model(X_new_tensor).cpu()
df_new_vitrine["Hacked_pred"] = (new_preds >= 0.5).int().numpy().flatten()

df_new_final = df_new_vitrine[df_new_vitrine["Hacked_pred"] == 1].copy()
df_new_final.rename(columns={"IdSubscriber": "Id", "Hacked_pred": "Hacked"}, inplace=True)
desired_cols_new = ["Id", "UID", "Type", "IdPlan", "TurnOn", "Hacked", "Traffic"]
missing_cols_new = [col for col in desired_cols_new if col not in df_new_final.columns]
if missing_cols_new:
    print("В итоговом DataFrame отсутствуют столбцы:", missing_cols_new)
else:
    df_new_final = df_new_final[desired_cols_new].drop_duplicates(subset=desired_cols_new)
    new_output_filename = os.path.join(new_folder, "RESULT.csv")
    df_new_final.to_csv(new_output_filename, index=False)
    print(f"\nИтоговый файл с предсказаниями сохранён как {new_output_filename}")





