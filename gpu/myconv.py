import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here. 

        # Precompute output size
        self.out_h = (H + 2 * padding - kernel_size) // stride + 1 
        self.out_w = (W + 2 * padding - kernel_size) // stride + 1 

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        

    def im2col_manual(self, x):
        N = x.shape[0]        # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))

        patches = []
        for kh in range(KH):
            for kw in range(KW):
                patch = x_pad[:, :, kh : kh + out_h * S : S, kw : kw + out_w * S : S]
                patches.append(patch)

        stacked = torch.stack(patches, dim=2)
        permuted = stacked.permute(0, 3, 4, 1, 2)
        cols = permuted.reshape(N, out_h * out_w, C * KH * KW)
        return cols

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        KH = KW = self.kernel_size

        # 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x) 

        # 2) flatten self.weight into shape (C_out, C*KH*KW).
        flattened_weight = self.weight.reshape(C_out, self.in_channels * KH * KW)

        # 3) perform tiled matmul after required reshaping is done.
        L = cols.shape[1]
        K = cols.shape[2]
        tile = 16
        weight_t = flattened_weight.T
        out = torch.zeros(N, L, C_out, device=cols.device, dtype=cols.dtype)
        for i in range(0, L, tile):
            i_end = min(i + tile, L)
            for j in range(0, C_out, tile):
                j_end = min(j + tile, C_out)
                acc = torch.zeros(N, i_end - i, j_end - j, device=cols.device, dtype=cols.dtype)
                for k in range(0, K, tile):
                    k_end = min(k + tile, K)
                    acc = acc + torch.matmul(cols[:, i:i_end, k:k_end], weight_t[k:k_end, j:j_end])
                out[:, i:i_end, j:j_end] = acc

        # 4) Add bias.
        out = out + self.bias

        # 5) reshape output into shape (N, C_out, out_h, out_w).
        out = out.reshape(N, self.out_h, self.out_w, C_out)
        out = out.permute(0, 3, 1, 2)
        
        return out

    def forward(self, x):
        return self.conv2d_manual(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 4, 22, 22
    device = "cuda"
    x = torch.randn(N, C, H, W, device=device)
    out_channels=8
    kernel_size=7
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(device)
    out = model(x)

    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("conv"):
            out = model(x)
        torch.cuda.synchronize()
    prof.export_chrome_trace("pytorch_trace.json")
