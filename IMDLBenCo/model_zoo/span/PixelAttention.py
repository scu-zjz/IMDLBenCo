import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelAttention(nn.Module):
    def __init__(self, in_channels=3, kernel_range=[3, 3], shift=1, use_bn=False, use_res=False):
        super(PixelAttention, self).__init__()
        self.in_channels = in_channels
        self.kernel_range = kernel_range
        self.shift = shift
        self.use_bn = use_bn
        self.use_res = use_res

        n_p = kernel_range[0] * kernel_range[1]
        self.k_p = nn.Conv2d(in_channels, in_channels * n_p, kernel_size=1)
        self.v_p = nn.Conv2d(in_channels, in_channels * n_p, kernel_size=1)
        self.q_p = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.ff1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.ff2 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1)
        self.ff3 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1)
        
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        h_half = self.kernel_range[0] // 2
        w_half = self.kernel_range[1] // 2
        s = x.shape
        D = s[1]
        # print("D:", D)
        x_k = self.k_p(x)
        x_v = self.v_p(x)
        x_q = self.q_p(x)
        # print("x_k.shape: ", x_k.shape)
        paddings = (w_half * self.shift, w_half * self.shift, h_half * self.shift, h_half * self.shift)
        # print("paddings: ", paddings)
        x_k = F.pad(x_k, paddings, "constant", 0)
        x_v = F.pad(x_v, paddings, "constant", 0)
        mask_x = torch.ones(s[0], 1, s[2], s[3], device=x.device)
        # print("x_k.shape: ", x_k.shape)
        # print("mask_x.shape: ", mask_x.shape)
        mask_pad = F.pad(mask_x, paddings, "constant", 0)
        
        k_ls = []
        v_ls = []
        masks = []
        
        c_x, c_y = h_half * self.shift, w_half * self.shift
        layer = 0
        # print("s[2]: ", s[2])
        # print("s[3]: ", s[3])
        for i in range(-h_half, h_half + 1):
            for j in range(-w_half, w_half + 1):
                
                k_t = x_k[:, layer*D:(layer+1)*D, c_x + i * self.shift:c_x + i * self.shift + s[2], c_y + j * self.shift:c_y + j * self.shift + s[3]]
                k_ls.append(k_t)
                v_t = x_v[:, layer*D:(layer+1)*D, c_x + i * self.shift:c_x + i * self.shift + s[2], c_y + j * self.shift:c_y + j * self.shift + s[3]]
                v_ls.append(v_t)
                # print("mask_pad.shape:", mask_pad.shape)
                _m = mask_pad[:, :, c_x + i * self.shift:c_x + i * self.shift + s[2], c_y + j * self.shift:c_y + j * self.shift + s[3]]
                # print("v_t: ", v_t.shape)
                # print("_m: ", _m.shape)
                masks.append(_m)
                layer += 1
        # print("mask: ", masks[0].shape)
        m_stack = torch.stack(masks, dim=1)
        m_vec = m_stack.view(s[0] * s[2] * s[3], self.kernel_range[0] * self.kernel_range[1], 1)
        # print("len(k_ls): ", len(k_ls))
        # print("k_ls[0]: ", k_ls[0].shape)
        k_stack = torch.stack(k_ls, dim=1)
        v_stack = torch.stack(v_ls, dim=1)
        # print("k_stack: ", k_stack.shape)
        k = k_stack.view(s[0] * s[2] * s[3] , self.kernel_range[0] * self.kernel_range[1], D)
        v = v_stack.view(s[0] * s[2] * s[3] , self.kernel_range[0] * self.kernel_range[1], D)
        q = x_q.view(s[0] * s[2] * s[3], 1, D)
        
        alpha = F.softmax(torch.matmul(k, q.transpose(-1, -2)) * m_vec / 8, dim=1)
        __res = torch.matmul(alpha.transpose(-1, -2), v)
        _res = __res.view(s[0], D, s[2], s[3])
        
        if self.use_res:
            t = x + _res
        else:
            t = _res
        
        if self.use_bn:
            t = self.bn1(t)
        
        _t = t
        t = F.relu(self.ff1(t))
        t = F.relu(self.ff2(t))
        t = F.relu(self.ff3(t))
        
        if self.use_res:
            t = _t + t
        
        if self.use_bn:
            res = self.bn2(t)
        else:
            res = t
        
        return res

# 测试 PixelAttention 类
if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 64, 64)  # (batch_size, in_channels, height, width)
    pixel_attention = PixelAttention(3)
    output = pixel_attention(input_tensor)
    print("output:", output.shape)
