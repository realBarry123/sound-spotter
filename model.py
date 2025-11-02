import torch
import torch.nn as nn

F = 128
B = 1

class SoundSpotter(nn.Module):
    def __init__(self):
        super().__init__()
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

    def to_query(self, short):
        # (B, 1, F, T)
        short = self.short_conv1(short)
        short = self.short_conv2(short)
        # (B, 16, F, T)
        short = short.mean(dim=-1, keepdim=False)
        # (B, 16, F)
        short = short.permute(0, 2, 1)
        # (B, F, T=16)
        # short = short.squeeze(0)
        return short
    
    def conv_heatmap(self, x, query):
        heatmaps = []
        for i in range(B):
            kernel = query[i].unsqueeze(0)
            heatmap = torch.nn.functional.conv1d(x[i:i+1], kernel)
            heatmaps.append(heatmap)
        heatmaps = torch.cat(heatmaps, dim=0)
        return heatmaps

    def forward(self, x, short):
        # (B x 1 x F x T)
        x = self.long_conv1(x)
        x = self.long_conv2(x)
        print("x: " + str(x.shape))

        query = self.to_query(short)
        print("query: " + str(query.shape))
        x = x.squeeze(1)
        print("x: " + str(x.shape))
        
        # (B x 1 x F x T)
        # heatmap = torch.nn.functional.conv2d(x, query)
        heatmap = self.conv_heatmap(x, query)
        
        # (B x 1 x T)
        count = heatmap.squeeze(1).sum(dim=-1, keepdim=False)
        # (B x 1)
        return count

def test_model_shape():
    x = torch.zeros((B, 1, F, 49298))
    short = torch.zeros((B, 1, F, 100))
    model = SoundSpotter()
    y = model(x, short)
    print(y.shape)

test_model_shape()