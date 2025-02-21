import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Giả sử bạn đã có dataset và dataloader
class HeartBeatDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Tải model pre-trained
model_path_ecg = "/Model_files/ECGPT_560k_iters.pth"
model = torch.load(model_path_ecg)

# Thêm lớp phân loại mới
num_classes = 4  # 4 loại nhịp tim
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Định nghĩa loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện model
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

# Giả sử bạn đã có data và labels
# data = ...  # Dữ liệu nhịp tim
# labels = ...  # Nhãn tương ứng (0: N, 1: S, 2: V, 3: F)
path_save = '/Data/Data_ECG/'
all_windows = np.load(path_save + 'all_windows.npy')
all_labels = np.load(path_save + 'all_labels.npy')

# Tạo dataset và dataloader
dataset = HeartBeatDataset(all_windows, all_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Huấn luyện model
train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# Lưu model đã huấn luyện
torch.save(model.state_dict(), 'finetuned_heartbeat_model.pth')