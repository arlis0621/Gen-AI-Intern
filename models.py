
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LATENT_DIM, SST_TARGET_SIZE, SST_PAD
# from geomdl import fitting, BSpline, knotvector

# Weight init (DCGAN-style)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        


def normalize_to_tanh(SST):
    mn, mx = SST.min(), SST.max()
    print(f"[normalize_to_tanh] before: min={mn:.3f}, max={mx:.3f}")
    if mx > mn:
        rst = 2.0 * (SST - mn) / (mx - mn) - 1.0
    else:
        rst = SST
    print(f"[normalize_to_tanh] after normalization")
    return rst




class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # 1) project & reshape
        self.fc  = nn.Linear(latent_dim, 640 * 2 * 2)
        self.bn0 = nn.BatchNorm2d(640)

        # 2) five ConvTranspose blocks
        #    Conv1: 640→384, kernel=4, stride=(2,4), pad=1  → out before crop: (B,384, 4, 6)
        self.deconv1 = nn.ConvTranspose2d(640, 384, kernel_size=4,stride=(2,4), padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(384)

        #    Conv2: 384→256, kernel=4, stride=(1,2), pad=1 → out before crop: (B,256, 5,12)
        self.deconv2 = nn.ConvTranspose2d(384, 256, kernel_size=4,stride=(1,2), padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(256)

        #    Conv3: 256→128, kernel=4, stride=2,    pad=1 → out before crop: (B,128, 8,16)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4,stride=2, padding=1, bias=False)
        self.bn3      = nn.BatchNorm2d(128)

        #    Conv4: 128→ 64, kernel=4, stride=2,    pad=1 → out before crop: (B, 64,14,30)
        self.deconv4 = nn.ConvTranspose2d(128,  64, kernel_size=4,stride=2, padding=1, bias=False)
        self.bn4      = nn.BatchNorm2d(64)

        #    Conv5:  64→  3, kernel=4, stride=2,    pad=1 → out before crop: (B,  3,26,58)
        self.deconv5 = nn.ConvTranspose2d(64,    3, kernel_size=4,stride=2, padding=1, bias=False)

    def _center_crop(self, x, th, tw):
        # assume x is (B,C,H,W); crop each to (th,tw) centered
        _, _, H, W = x.shape
        sh = (H - th) // 2
        sw = (W - tw) // 2
        return x[:, :, sh:sh+th, sw:sw+tw]

    def forward(self, z):
        B = z.size(0)

        # project & reshape → (B,640,2,2)
        out = self.fc(z).view(B, 640, 2, 2)
        out = F.relu(self.bn0(out), inplace=True)

        #   → deconv1 → (B,384,4,6) directly
        out = self.deconv1(out)
        out = F.relu(self.bn1(out), inplace=True)
        # no need to crop; stride=(2,4), pad=1 already gives exactly 4×6

        #   → deconv2 → (B,256,5,12) → crop to (4,8)
        out = self.deconv2(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self._center_crop(out, 4, 8)

        #   → deconv3 → (B,128,8,16) → crop to (7,15)
        out = self.deconv3(out)
        out = F.relu(self.bn3(out), inplace=True)
        out = self._center_crop(out, 7, 15)

        #   → deconv4 → (B, 64,14,30) → crop to (13,29)
        out = self.deconv4(out)
        out = F.relu(self.bn4(out), inplace=True)
        out = self._center_crop(out, 13, 29)

        #   → deconv5 → (B,  3,26,58) → crop to (25,57) + Tanh
        out = self.deconv5(out)
        out = torch.tanh(out)
        out = self._center_crop(out, 25, 57)

        return out  # (B, 3, 25, 57)


import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super().__init__()
        # 50% dropout on the input SST
        self.dropout = nn.Dropout2d(0.5)
        # 1) 64×64 → 32×32
        self.conv1 = nn.Conv2d(input_channels, base_filters,   kernel_size=4, stride=2, padding=1, bias=False)
        # 2) 32×32 → 16×16  (with BatchNorm)
        self.conv2 = nn.Conv2d(base_filters,     base_filters*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(base_filters*2)
        # 3) 16×16 →  8× 8
        self.conv3 = nn.Conv2d(base_filters*2,   base_filters*4, kernel_size=4, stride=2, padding=1, bias=False)
        # 4)  8× 8 →  4× 4  (with BatchNorm)
        self.conv4 = nn.Conv2d(base_filters*4,   base_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(base_filters*8)
        # 5)  4× 4 →  1× 1
        self.conv5 = nn.Conv2d(base_filters*8, 1,               kernel_size=4, stride=1, padding=0, bias=False)

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.sig   = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)                       # (B,3,64,64)
        x = self.lrelu(self.conv1(x))             # → (B,64,32,32)
        x = self.lrelu(self.bn2(self.conv2(x)))   # → (B,128,16,16)
        x = self.lrelu(self.conv3(x))             # → (B,256, 8, 8)
        x = self.lrelu(self.bn4(self.conv4(x)))   # → (B,512, 4, 4)
        x = self.sig(self.conv5(x))               # → (B,  1, 1, 1)
        return x.view(-1)                         # → (B,)