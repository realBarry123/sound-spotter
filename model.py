import torch
import torch.nn as nn

class SoundSpotter(nn.Module):
    def __init__(self, F: int, B: int):
        super().__init__()
        self.F = F
        self.B = B
        self.short_conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=4, 
            kernel_size=(3, 3),
            padding="same"
        )

        self.short_conv2 = nn.Conv2d(
            in_channels=4, 
            out_channels=16, 
            kernel_size=(3, 3),
            padding="same"
        )

        self.long_conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=(5, 5),
            padding="same"
        )

        self.long_conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=1, 
            kernel_size=(5, 5),
            padding="same"
        )

        self.short_relu = nn.LeakyReLU()
        self.long_relu = nn.LeakyReLU()

        self.sigmoid = nn.Sigmoid()

    def to_query(self, short: torch.Tensor):
        # (B, 1, F, T)
        short = self.short_conv1(short)
        short = self.short_relu(short)
        short = self.short_conv2(short) # (B, 16, F, T)
        short = short.mean(dim=-1, keepdim=False) # (B, 16, F)
        short = short.permute(0, 2, 1) # (B, F, T=16)
        return short

    def forward(self, x: torch.Tensor, short: torch.Tensor):
        # (B, 1, F, T)
        x = self.long_conv1(x)
        x = self.long_relu(x)
        x = self.long_conv2(x)

        query = self.to_query(short)
        x = x.squeeze(1) # (B, F, T)

        x = torch.nn.functional.normalize(x, dim=1)
        query = torch.nn.functional.normalize(query, dim=1)

        heatmap_list = []
        for i in range(self.B):
            # x[i]: (F, T), query[i]: (F, K)
            conv_out = torch.nn.functional.conv1d(input=x[i:i+1], weight=query[i:i+1])
            heatmap_list.append(conv_out)
        
        # (B, 1, T')
        heatmap = torch.cat(heatmap_list, dim=0)
        heatmap = self.sigmoid(heatmap)
        count = heatmap.squeeze(1).sum(dim=-1, keepdim=False) # (B)
        return count

def test_model_shape():
    F = 128
    B = 4
    x = torch.zeros((B, 1, F, 49298))
    short = torch.zeros((B, 1, F, 100))
    model = SoundSpotter(F, B)
    y = model(x, short)
    print(y.shape)

test_model_shape()