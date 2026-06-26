import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.conv1 = nn.Conv1d(
            input_features,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            32,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,1)
        )

    def forward(self,x):

        x = x.permute(0,2,1)

        x = torch.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = torch.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = x.permute(0,2,1)

        x,_ = self.lstm(x)

        x = x[:,-1,:]

        x = self.fc(x)

        return x
