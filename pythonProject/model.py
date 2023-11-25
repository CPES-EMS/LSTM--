from torch import nn
# class MlpNet(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(MlpNet, self).__init__()
#
#         # self.mlp = nn.Sequential(
#         #     nn.Linear(input_size, 256),
#         #     nn.ReLU(),
#         #     nn.Linear(256, 512),
#         #     nn.ReLU(),
#         #     nn.Linear(512, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, output_size),
#         # )
#         self.mlp = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.LeakyReLU(),
#             nn.Linear(64, output_size),
#         )
#
#     def forward(self, x):
#         x = self.mlp(x)
#         x = torch.reshape(x, (-1, 4))
#         return x


class LstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, label_size):
        super(LstmNet, self).__init__()
        self.label_size = label_size
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
        )
        # 对LSTM层和全连接层的权重进行正交初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)


    def forward(self, x):
        x, _ = self.lstm(x)
        batch_size, seq_len, hid_dim = x.shape
        x = x.reshape(-1, hid_dim)
        x = self.linear(x)
        x = x.reshape(batch_size, seq_len, self.output_size)
        return x


# net = MlpNet(1, 4)
