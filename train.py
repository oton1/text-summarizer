import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.cuda

train_dataset_path = 'train.txt'
val_dataset_path = 'val.txt'
input_size = 512
hidden_size = 768
output_size = 12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()

class TextSummarizationDataset(Dataset):
    def __init__(self, data_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.inputs, self.targets = self._load_data(data_path)
        
    def _load_data(self, data_path):
        with open(data_path, "r") as f:
            lines = f.readlines()
        inputs = []
        targets = []
        for i, line in enumerate(lines):
            try:
                input_text, target_text = line.strip().split('_')
                inputs.append(input_text)
                targets.append(target_text)
            except ValueError:
                print(f"Error: line {i+1} '{line.strip()}' has an incorrect format")
        return inputs, targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        input_tokens = self.tokenizer.encode(input_text, add_special_tokens=True, max_length=512, truncation=True)
        target_tokens = self.tokenizer.encode(target_text, add_special_tokens=True, max_length=128, truncation=True)
        input_tensor = torch.tensor(input_tokens)
        target_tensor = torch.tensor(target_tokens)
        input_shape = input_tensor.shape
        target_shape = target_tensor.shape
        print(f"input shape:{input_shape}, target shape: {target_shape}")
        return input_tensor, target_tensor

class TextSummarizationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs):
        encoded = self.encoder(inputs)[0]  # output of shape (batch_size, seq_len, hidden_size)
        summarized = self.decoder(encoded[:, 0, :])  # summary is the first token of the input
        return summarized

def get_data_loaders(train_dataset_path, val_dataset_path, batch_size):
    train_dataset = TextSummarizationDataset(train_dataset_path)
    val_dataset = TextSummarizationDataset(val_dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def train(model, train_dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_dataloader)

def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)

train_loader, val_loader = get_data_loaders(train_dataset_path, val_dataset_path, batch_size=12)
model = TextSummarizationModel(input_size, hidden_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, loss_fn, device)
    val_loss = evaluate(model, val_loader, loss_fn, device)

    print(f' Epoch {epoch}: val_loss={val_loss:.4f}')