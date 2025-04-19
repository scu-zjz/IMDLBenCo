import torch
import torch.nn as nn
import torch.fft
import math
class LowDctFrequencyExtractor(nn.Module):
    def __init__(self, alpha=0.95):
        super(LowDctFrequencyExtractor, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None

    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    def high_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        mask[-alpha_h:, -alpha_w:] = 0

        return x * mask

    def forward(self, x):
        xq = self.dct_2d(x)
        xq_high = self.high_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals)
        return xh