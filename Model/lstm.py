import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


# LSTM 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=100, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm1 = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.35)
        self.lstm2 = nn.LSTM(hidden_layer_size, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.35)
        self.lstm3 = nn.LSTM(50, 50, batch_first=True)
        self.dropout3 = nn.Dropout(0.35)
        self.linear = nn.Linear(50, output_size)

    def forward(self, x):

        h0_lstm1 = torch.zeros(1, x.size(0), self.hidden_layer_size).to(x.device)
        c0_lstm1 = torch.zeros(1, x.size(0), self.hidden_layer_size).to(x.device)
        e0_lstm1 = torch.zeros(1, x.size(0), self.hidden_layer_size).to(x.device)


        h0_lstm2 = torch.zeros(1, x.size(0), 50).to(x.device)
        c0_lstm2 = torch.zeros(1, x.size(0), 50).to(x.device)
        e0_lstm2 = torch.zeros(1, x.size(0), 50).to(x.device)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()

        x, _ = self.lstm1(x, (h0_lstm1, c0_lstm1))
        x = self.dropout1(x)
        x, _ = self.lstm2(x, (h0_lstm2, c0_lstm2))
        x = self.dropout2(x)
        x, _ = self.lstm3(x, (h0_lstm2, c0_lstm2))
        x = self.dropout3(x)
        x = self.linear(x)
        return x


# 定义RMSE损失函数
def RMSE_loss(output, target):
    return torch.sqrt(torch.mean((output - target) ** 2))


# 转换数据为适合 LSTM 的格式
def create_sequences(train_data, target_data, time_steps):
    train_sequences = []
    target_sequences = []
    for i in range(len(train_data) - time_steps):
        train_sequences.append(train_data[i:(i + time_steps)])
        target_sequences.append(target_data[i + time_steps])
    return np.array(train_sequences), np.array(target_sequences)


# 加载和预处理数据

# Data Processing Functions
def load_and_preprocess_data(folder_path, train_col_index, target_col_index, scaler_train, scaler_target):
    all_train_sequences = []
    all_target_sequences = []

    # Fit the scalers on the entire dataset
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            data = pd.read_csv(file_path)
            all_data.append(data)

    # Combine all dataframes into a single dataframe
    full_data = pd.concat(all_data, axis=0)

    # Fit the scalers on the entire dataset
    scaler_train.fit_transform(full_data.iloc[:, train_col_index].values.reshape(-1, 1))
    scaler_target.fit_transform(full_data.iloc[:, target_col_index].values.reshape(-1, 1))

    # Process each file using the fitted scalers
    for data in all_data:
        train_col = data.iloc[:, train_col_index]
        target_col = data.iloc[:, target_col_index]

        train_scaled = scaler_train.transform(train_col.values.reshape(-1, 1))
        target_scaled = scaler_target.transform(target_col.values.reshape(-1, 1))

        train_sequences, target_sequences = create_sequences(train_scaled, target_scaled, TIME_STEPS)
        all_train_sequences.append(train_sequences)
        all_target_sequences.append(target_sequences)

    return np.concatenate(all_train_sequences, axis=0), np.concatenate(all_target_sequences, axis=0)


# 主函数
def main():
    folder_path = 'data/2023_01_type_1'
    input_features = [1, 2, 3]  # TX, TY, TZ  的列索引
    target_features = [4, 5, 6]  # SX, SY, SZ 的列索引

    for input_idx, target_idx in zip(input_features, target_features):

        train_sequences, target_sequences = load_and_preprocess_data(folder_path, [input_idx], target_idx,
                                                                     MinMaxScaler(), MinMaxScaler())

        # 转换为 PyTorch 张量
        train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
        target_sequences = torch.tensor(target_sequences, dtype=torch.float32)

        # 创建 DataLoader
        train_data = TensorDataset(train_sequences, target_sequences)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        # 创建模型
        model = LSTMModel(1, output_size=1)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = RMSE_loss
        early_stopping = EarlyStopping(patience=5, verbose=True)

        # 训练模型
        model.train()
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            running_loss = 0.0

            # 使用 tqdm 创建进度条
            progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f'Epoch {epoch + 1}/{EPOCHS}')
            for i, (inputs, targets) in progress_bar:
                inputs, targets = inputs, targets

                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 更新进度条
                avg_loss = running_loss / i
                elapsed_time = time.time() - epoch_start_time
                eta = elapsed_time / i * (len(train_loader) - i)
                progress_bar.set_postfix(loss=avg_loss, ETA=f'{eta:.2f}s')

            # 在每个 epoch 结束后打印平均损失
            print(f'Epoch {epoch + 1} Average Loss: {running_loss / len(train_loader)}')
            scheduler.step()

            if early_stopping(running_loss, model):
                print("Early stopping.")
                break
        # 保存模型，文件名中包含目标列索引
        torch.save(model.state_dict(), f'lstm_model_input_{input_idx + 1}_target_{target_idx + 1}.pth')


if __name__ == "__main__":
    BATCH_SIZE = 64  # 批处理大小
    EPOCHS = 30  # 训练轮次
    LEARNING_RATE = 0.001  # 学习率
    TIME_STEPS = 1  # 时间步长
    INPUT_SIZE = 1  # 输入特征数（根据你的数据调整）
    OUTPUT_SIZE = 1  # 输出特征数（根据你的数据调整）

    main()
