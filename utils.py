import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
import random
import numpy as np
from typing import Iterable, Callable
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, train_loader, optimizer, criterion,scheduler=None):
    model.train(True)
    total_loss = 0.0
    total_correct = 0

    for inputs, labels in train_loader:

        inputs = inputs.float().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        loss.backward()

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
    # 更新学习率（如果提供了 scheduler）
    if scheduler is not None:
        scheduler.step()
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = total_correct.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc.item()

def valid(model, valid_loader, criterion):

    model.eval()
    total_loss = 0.0
    total_correct = 0
    true = (torch.tensor([])).to(device)
    pred = (torch.tensor([])).to(device)
    for inputs,  labels in valid_loader:

        inputs = inputs.float().to(device)
        # img = img.float().to(device)
        # inputs = inputs.float().to(device)

        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
        true = torch.cat((true, labels.data), 0)
        pred = torch.cat((pred, predictions), 0)
    epoch_loss = total_loss / len(valid_loader.dataset)
    epoch_acc = total_correct.double() / len(valid_loader.dataset)
    return epoch_loss, epoch_acc.item(), true, pred

def calculate_metrics_multilabel(y_true, y_pred):
    # 将张量转换为 NumPy 数组
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # 计算 Precision、Recall 和 F1-Score
    precision = precision_score(y_true_np, y_pred_np, average='macro')
    recall = recall_score(y_true_np, y_pred_np, average='macro')
    f1 = f1_score(y_true_np, y_pred_np, average='macro')

    return precision, recall, f1

def valid(model, valid_loader, criterion):

    model.eval()
    total_loss = 0.0
    total_correct = 0
    true = (torch.tensor([])).to(device)
    pred = (torch.tensor([])).to(device)
    for inputs,  labels in valid_loader:

        inputs = inputs.float().to(device)

        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
        true = torch.cat((true, labels.data), 0)
        pred = torch.cat((pred, predictions), 0)
    epoch_loss = total_loss / len(valid_loader.dataset)
    valid_true = true.cpu()
    valid_pred = pred.cpu()
    cm = confusion_matrix(valid_true, valid_pred)
    epoch_acc = total_correct.double() / len(valid_loader.dataset)
    precision, recall, f1_score = calculate_metrics_multilabel(true, pred)

    return epoch_loss, epoch_acc.item(), cm, precision, recall, f1_score
