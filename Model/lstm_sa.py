import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class Config:
    input_size = 3
    hidden_size = 256  # Hidden layer size
    num_layers = 4  # Number of LSTM layers
    output_size = 3
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.00001
    model_save_path = 'lstm_selfatttneion_model_simple_0613.pth'
    scaler = StandardScaler()
    patience = 10
    factor = 0.1
    max_epochs_without_improvement = 5
    plots_save_path = 'simple_training_plots_0613'


config = Config()
os.makedirs(config.plots_save_path, exist_ok=True)

filepaths = [
    '../c_interpolated_data_202303_b_d.csv',
    '../c_interpolated_data_202302_b_d.csv',
    '../c_interpolated_data_202301_b_d.csv',
    '../c_interpolated_data_202305_b_d.csv',
    '../c_interpolated_data_202304_b_d.csv'
]


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.seq_len = 1  # Setting sequence length to 1 for simplicity

    def __len__(self):
        return len(self.X) - (self.seq_len - 1)

    def __getitem__(self, idx):
        return self.X[idx:idx + self.seq_len].reshape(self.seq_len, -1), self.y[idx + self.seq_len - 1]


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, 1)

    def forward(self, outputs):
        # outputs shape: (batch_size, seq_len, hidden_size)
        attn_weights = F.softmax(self.attn(outputs).squeeze(2), dim=1)
        # attn_weights shape: (batch_size, seq_len)
        new_hidden_state = torch.bmm(outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        # new_hidden_state shape: (batch_size, hidden_size)
        return new_hidden_state, attn_weights


class LSTMSelfAttention(nn.Module):
    def __init__(self):
        super(LSTMSelfAttention, self).__init__()
        self.lstm = nn.LSTM(config.input_size, config.hidden_size, config.num_layers, batch_first=True)
        self.attention = Attention(config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        attn_out, attn_weights = self.attention(lstm_out)  # attn_out shape: (batch_size, hidden_size)
        out = self.fc(attn_out)  # Final output
        return out


def load_data(filepath):
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    df = df.head(int(len(df) * 1))
    features = df[['TRP_PhaPos_X', 'TRP_PhaPos_Y', 'TRP_PhaPos_Z']].values
    labels = df[['SDP_PhaPos_X', 'SDP_PhaPos_Y', 'SDP_PhaPos_Z']].values
    features_scaled = config.scaler.fit_transform(features)
    labels_scaled = config.scaler.transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_scaled, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_mse, total_mae, total_r2 = 0, 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            preds = predictions.detach().numpy()
            targets = y_batch.detach().numpy()
            total_mse += mean_squared_error(targets, preds)
            total_mae += mean_absolute_error(targets, preds)
            total_r2 += r2_score(targets, preds)

    avg_loss = total_loss / len(loader)
    avg_mse = total_mse / len(loader)
    avg_mae = total_mae / len(loader)
    avg_r2 = total_r2 / len(loader)
    return avg_loss, avg_mse, avg_mae, avg_r2


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, config, filepath):
    best_loss = float('inf')
    epochs_without_improvement = 0
    results_df = pd.DataFrame(
        columns=['Epoch', 'Training Loss', 'Validation Loss', 'Training MSE', 'Validation MSE', 'Training MAE',
                 'Validation MAE', 'Training R²', 'Validation R²'])

    # 为了绘图初始化指标的历史记录
    train_loss_history, val_loss_history = [], []
    train_mse_history, val_mse_history = [], []
    train_mae_history, val_mae_history = [], []
    train_r2_history, val_r2_history = [], []
    for epoch in range(config.num_epochs):
        model.train()
        total_loss, total_mse, total_mae, total_r2 = 0, 0, 0, 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = predictions.detach().numpy()
            targets = y_batch.detach().numpy()
            total_mse += mean_squared_error(targets, preds)
            total_mae += mean_absolute_error(targets, preds)
            total_r2 += r2_score(targets, preds)

        avg_loss = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        avg_mae = total_mae / len(train_loader)
        avg_r2 = total_r2 / len(train_loader)

        # 记录训练集的指标历史
        train_loss_history.append(avg_loss)
        train_mse_history.append(avg_mse)
        train_mae_history.append(avg_mae)
        train_r2_history.append(avg_r2)

        val_loss, val_mse, val_mae, val_r2 = evaluate(model, val_loader, criterion)

        # 记录验证集的指标历史
        val_loss_history.append(val_loss)
        val_mse_history.append(val_mse)
        val_mae_history.append(val_mae)
        val_r2_history.append(val_r2)

        scheduler.step(val_loss)
        savemodelsname = str(filepath) + str(epoch) + config.model_save_path
        torch.save(model.state_dict(), savemodelsname)
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.model_save_path)
            # print(f"Epoch {epoch + 1}: Model improved and saved with validation loss = {best_loss}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.max_epochs_without_improvement:
                print(f"No improvement for {config.max_epochs_without_improvement} epochs, stopping training.")
                break

        results_df = results_df.append({'Epoch': epoch + 1, 'Training Loss': avg_loss, 'Validation Loss': val_loss,
                                        'Training MSE': avg_mse, 'Validation MSE': val_mse,
                                        'Training MAE': avg_mae, 'Validation MAE': val_mae,
                                        'Training R²': avg_r2, 'Validation R²': val_r2}, ignore_index=True)
        print(
            f"Epoch{epoch + 1}:Training loss={avg_loss},Validation loss ={val_loss}")
        print(
            f"Epoch{epoch + 1}:MSE ={avg_mse},MAE={avg_mae},R² = {avg_r2}")
        results_csv_path = str(filepath) + str(epoch) + "training_results_0402.csv"
        results_df.to_csv(os.path.join(config.plots_save_path, results_csv_path), index=False)
        # 在训练结束后保存结果和绘图
        results_df.to_csv(os.path.join(config.plots_save_path, results_csv_path), index=False)
        plot_training_progress(config.plots_save_path, train_loss_history, val_loss_history, train_mse_history,
                               val_mse_history, train_mae_history, val_mae_history, train_r2_history, val_r2_history,
                               epoch, filepath)


def plot_training_progress(save_path, train_loss, val_loss, train_mse, val_mse, train_mae, val_mae, train_r2, val_r2,
                           epoch, filepath):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(15, 10))

    # 绘制损失图
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制MSE图
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_mse, 'b-', label='Training MSE')
    plt.plot(epochs, val_mse, 'r--', label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    # 绘制MAE图
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_mae, 'b-', label='Training MAE')
    plt.plot(epochs, val_mae, 'r--', label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # 绘制R²图
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_r2, 'b-', label='Training R²')
    plt.plot(epochs, val_r2, 'r--', label='Validation R²')
    plt.title('Training and Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()

    save_pathsss = str(filepath) + str(epoch) + 'training_metrics_progress.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_pathsss))
    plt.close()


def main():
    model = LSTMSelfAttention()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.patience, factor=config.factor,
                                                     verbose=True)

    for filepath in filepaths:
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(filepath)
        train_dataset = CustomDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = CustomDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        print(f"Starting training with {filepath}")
        train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, config, filepath)
    print("Final model training complete.")


if __name__ == "__main__":
    main()
