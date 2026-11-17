import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DATA_DIR

WEIGHTS = DATA_DIR / "models" / "NAFNet-GoPro-width64.pth"
GDRIVE_ID = "1S0PVRbyTakYY9a82kujgZLbMihfNBLfC"
GOPRO_CFG = dict(width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
TRAIN_SIZE = (1, 3, 256, 256)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw, 1, bias=True)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw, bias=True)
        self.conv3 = nn.Conv2d(dw // 2, c, 1, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1, bias=True))
        self.sg = SimpleGate()
        ffn = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1, bias=True)
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=(), dec_blk_nums=()):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        self.encoders, self.decoders, self.ups, self.downs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        _, _, H, W = inp.shape
        ph = (self.padder_size - H % self.padder_size) % self.padder_size
        pw = (self.padder_size - W % self.padder_size) % self.padder_size
        inp = F.pad(inp, (0, pw, 0, ph))
        x = self.intro(inp)
        encs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = dec(up(x) + skip)
        x = self.ending(x) + inp
        return x[:, :, :H, :W]


class LocalAvgPool2d(nn.Module):
    """Test-time local converter (TLC, Chu et al. 2021): average over a window scaled from
    the training crop instead of the whole image. Same as the official AvgPool2d (fast_imp=False)."""

    def __init__(self, base_size, train_size):
        super().__init__()
        self.base_size, self.train_size, self.kernel_size = base_size, train_size, None

    def forward(self, x):
        if self.kernel_size is None:
            self.kernel_size = [x.shape[2] * self.base_size[0] // self.train_size[-2],
                                x.shape[3] * self.base_size[1] // self.train_size[-1]]
        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)
        n, c, h, w = x.shape
        s = F.pad(x.cumsum(dim=-1).cumsum(dim=-2), (1, 0, 1, 0))
        k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
        out = (s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2]) / (k1 * k2)
        _h, _w = out.shape[2:]
        return F.pad(out, ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2), mode="replicate")


def _to_local(module, base_size, train_size):
    for name, m in module.named_children():
        if len(list(m.children())) > 0:
            _to_local(m, base_size, train_size)
        if isinstance(m, nn.AdaptiveAvgPool2d):
            setattr(module, name, LocalAvgPool2d(base_size, train_size))


def ensure_weights():
    """Download the official weights with gdown if they are missing."""
    if WEIGHTS.exists():
        return WEIGHTS
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
        gdown.download(id=GDRIVE_ID, output=str(WEIGHTS), quiet=False)
    except Exception as e:
        raise FileNotFoundError(
            f"NAFNet weights missing ({e}). Install gdown (pip install gdown) or download "
            f"https://drive.google.com/file/d/{GDRIVE_ID}/view by hand and save it as {WEIGHTS}")
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"download failed; save the file by hand as {WEIGHTS}")
    return WEIGHTS


def load_nafnet(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = NAFNet(**GOPRO_CFG)
    state = torch.load(ensure_weights(), map_location="cpu", weights_only=True)
    state = state.get("params", state)
    net.load_state_dict(state, strict=True)
    base = (int(TRAIN_SIZE[-2] * 1.5), int(TRAIN_SIZE[-1] * 1.5))
    _to_local(net, base, TRAIN_SIZE)
    fix_local_windows(net)
    return net.eval().to(device), device


@torch.no_grad()
def fix_local_windows(net):
    """As in the official convert(): one pass at the training size fixes every pooling window
    (1.5 x the feature size seen in training). Larger inputs then pool locally."""
    net.eval()
    net(torch.rand(TRAIN_SIZE, device=next(net.parameters()).device))


@torch.no_grad()
def nafnet_deblur(net, img, device):
    """img: H x W (gray, copied to 3 channels; output = channel mean) or H x W x 3, floats in [0, 1]."""
    gray = img.ndim == 2
    x = np.repeat(img[..., None], 3, axis=2) if gray else img
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()[None].to(device)
    out = net(t.clamp(0, 1))[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return out.mean(axis=2) if gray else out
