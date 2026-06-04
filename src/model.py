
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class MulticlassClassificationWithAttentionHead(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassificationWithAttentionHead, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 1024)
        self.layer_2 = nn.Linear(1024, 512)
        self.layer_3 = nn.Linear(512, 256)
        self.layer_5 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        
        self.multihead_attention_before = nn.MultiheadAttention(embed_dim=1024, num_heads=2)
        self.multihead_attention_after = nn.MultiheadAttention(embed_dim=64, num_heads=2)

        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.batchnorm5 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = x.unsqueeze(0)  # Add batch dimension [1, L, N] because multihead attention expects it
        
        # Apply attention
        x, _ = self.multihead_attention_before(x, x, x)
        x = x.squeeze(0)  # Remove batch dimension

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)

        x = self.layer_5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = x.unsqueeze(0)  

        # Apply attention
        x, _ = self.multihead_attention_after(x, x, x)
        x = x.squeeze(0)  

        x = self.layer_out(x)

        return x

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def train_model(X, y, num_classes, epochs=10, batch_size=1048, learning_rate=5e-5, device="cpu"):
    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1234)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=1234)

    # Convert to tensors
    train_dataset = ClassifierDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val.values).float(), torch.from_numpy(y_val.values).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test.values).float(), torch.from_numpy(y_test.values).long())

    # Weighted Sampler logic to handle class imbalance
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)
    target_list = torch.tensor(target_list)
    
    class_count = [i for i in pd.Series(y_train).value_counts().sort_index()] # Ensure sorted by class index
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    
    class_weights_all = class_weights[target_list]
    
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=weighted_sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    model = MulticlassClassificationWithAttentionHead(num_feature=X.shape[1], num_class=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    accuracy_stats = {'train': [], 'val': []}
    loss_stats = {'train': [], 'val': []}

    print(f"Begin training on {device} for {epochs} epochs...")
    
    start_total = time.time()
    for e in range(1, epochs + 1):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_train_pred = model(X_train_batch)
            
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            
        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        
        print(f'Epoch {e:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

    # Evaluation on Test Set
    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
    return accuracy_stats, loss_stats, y_test, y_pred_list
