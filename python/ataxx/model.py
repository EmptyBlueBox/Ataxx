import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

PLANES, KINDS, SQ = 14, 17, 49


def pooled(x):
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)


class Linear(nn.Linear):
    def __init__(self, inputs, outputs, quant):
        super().__init__(inputs, outputs)
        self.quant = quant
        if quant:
            self.register_buffer("lo", torch.zeros(()))
            self.register_buffer("hi", torch.zeros(()))

    def forward(self, x):
        if not self.quant:
            return super().forward(x)
        x = x.float()
        if self.training:
            v = x.detach().flatten()
            v = v[torch.randint(0, len(v), (8192,), device=v.device)].sort().values
            lo, hi = v[1], v[-2]
            first = self.hi == self.lo
            self.lo.copy_(torch.where(first, lo, self.lo.lerp(lo, 0.01)))
            self.hi.copy_(torch.where(first, hi, self.hi.lerp(hi, 0.01)))
        s = (self.hi - self.lo) / 255
        x = x + (((x - self.lo) / s).round().clamp(0, 255) * s + self.lo - x).detach()
        w = self.weight
        scale = w.abs().amax(1, keepdim=True) / 127
        return F.linear(x, w + ((w / scale).round() * scale - w).detach(), self.bias)


class Attention(nn.Module):
    def __init__(self, c, heads, ratio, gelu, quant, key, value, soft):
        super().__init__()
        self.heads, self.key, self.value, self.soft = heads, key or c // heads, value or c // heads, soft
        self.norm1 = nn.LayerNorm(c)
        self.qkv = Linear(c, heads * (2 * self.key + self.value), quant)
        self.proj = Linear(heads * self.value, c, quant)
        self.bias = nn.Parameter(torch.zeros(heads, SQ, SQ))
        self.norm2 = nn.LayerNorm(c)
        self.mlp = nn.Sequential(Linear(c, ratio * c, quant), nn.GELU() if gelu else nn.ReLU(), Linear(ratio * c, c, quant))

    def forward(self, x):
        n = len(x)
        h, k = self.heads, self.key
        q, key, v = self.qkv(self.norm1(x)).split([h * k, h * k, h * self.value], 2)
        q, key, v = (t.view(n, SQ, h, -1).transpose(1, 2) for t in (q, key, v))
        a = q @ key.transpose(-1, -2) * k ** -0.5 + self.bias
        a = a.softmax(-1) if self.soft else F.relu(a) / SQ
        x = x + self.proj((a @ v).transpose(1, 2).reshape(n, SQ, -1))
        return x + self.mlp(self.norm2(x))


class Tokens(nn.Module):
    def __init__(self, c, blocks, heads, ratio, gelu, quant, key, value, soft):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(SQ, c))
        self.blocks = nn.Sequential(*[Attention(c, heads, ratio, gelu, quant, key, value, soft) for _ in range(blocks)])
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        n, c = x.shape[:2]
        return self.norm(self.blocks(x.flatten(2).transpose(1, 2) + self.position)).transpose(1, 2).reshape(n, c, 7, 7)


class Net(nn.Module):
    def __init__(self, blocks=6, channels=64, heads=4, ratio=2, gelu=True, quant=False, key=0, value=0, soft=True):
        super().__init__()
        c = channels
        self.config = dict(blocks=blocks, channels=channels, heads=heads, ratio=ratio, gelu=gelu, quant=quant, key=key, value=value, soft=soft)
        self.stem = nn.Conv2d(PLANES, c, 3, padding=1)
        self.blocks = Tokens(c, blocks, heads, ratio, gelu, quant, key, value, soft)
        self.policy1 = nn.Conv2d(c, 32, 1)
        self.policy_pool = nn.Linear(2 * c, 32)
        self.policy2 = nn.Conv2d(32, KINDS, 1)
        self.value1 = nn.Conv2d(c, 32, 1)
        self.value2 = nn.Linear(64, 64)
        self.value3 = nn.Linear(64, 4)
        self.ownership = nn.Conv2d(c, 1, 1)

    def trunk(self, x):
        return self.blocks(self.stem(x))

    def heads(self, t):
        p = F.relu(self.policy1(t) + self.policy_pool(pooled(t))[:, :, None, None])
        policy = self.policy2(p).flatten(1)
        value = self.value3(F.relu(self.value2(pooled(F.relu(self.value1(t))))))
        return policy, value

    def forward(self, x):
        return self.heads(self.trunk(x))

    def train_forward(self, x):
        t = self.trunk(x)
        return (*self.heads(t), self.ownership(t).flatten(1))


def save(net, path):
    torch.save(dict(config=net.config, state=net.state_dict()), path)


def load(path, device="cpu"):
    data = torch.load(path, map_location=device)
    net = Net(**data["config"]).to(device)
    net.load_state_dict(data["state"])
    return net.eval()


def export(net, path):
    state = net.cpu().float().state_dict()
    raw = lambda *keys: b"".join(state[k].numpy().astype(np.float32).tobytes() for k in keys)

    def dense(name):
        w, b = state[name + ".weight"], state[name + ".bias"]
        if not net.config["quant"]:
            return raw(name + ".weight", name + ".bias")
        lo, s = state[name + ".lo"], (state[name + ".hi"] - state[name + ".lo"]) / 255
        scale = w.abs().amax(1, keepdim=True) / 127
        q = (w / scale).round()
        return (torch.stack([lo, s]).numpy().astype(np.float32).tobytes() + q.numpy().astype(np.int8).tobytes()
                + (s * scale[:, 0]).numpy().astype(np.float32).tobytes() + (b + lo * scale[:, 0] * q.sum(1)).numpy().astype(np.float32).tobytes())

    with open(path, "wb") as f:
        f.write(np.array([net.config[k] for k in ("blocks", "channels", "heads", "ratio", "gelu", "quant", "key", "value", "soft")], np.int32).tobytes())
        f.write(raw("stem.weight", "stem.bias", "blocks.position"))
        for i in range(net.config["blocks"]):
            k = f"blocks.blocks.{i}."
            f.write(raw(k + "norm1.weight", k + "norm1.bias") + dense(k + "qkv") + dense(k + "proj") + raw(k + "bias", k + "norm2.weight", k + "norm2.bias")
                    + dense(k + "mlp.0") + dense(k + "mlp.2"))
        f.write(raw("blocks.norm.weight", "blocks.norm.bias", *[f"{h}.{k}" for h in ("policy1", "policy_pool", "policy2", "value1", "value2", "value3") for k in ("weight", "bias")]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export(load(args.pt), args.out)


if __name__ == "__main__":
    main()
