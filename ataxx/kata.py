import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper

from ataxx.model import NMOVES, SHIFTS

DX = [-2, -1, 0, 1, 2, -2, 2, -2, 2, -2, 2, -2, -1, 0, 1, 2]
DY = [-2, -2, -2, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 2, 2, 2]


def rms(x, w):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * w


def gpool(x, value):
    m = x.mean((2, 3))
    return torch.cat([m, m * -0.7, m * (0.39) if value else x.amax((2, 3))], 1)


class Block(nn.Module):
    def __init__(self, c, heads, ffn):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.Parameter(torch.ones(c)), nn.Parameter(torch.ones(c))
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.out = nn.Linear(c, c, bias=False)
        self.ffn1 = nn.Linear(c, 2 * ffn, bias=False)
        self.ffn2 = nn.Linear(ffn, c, bias=False)

    def forward(self, x, cos, sin):
        n, s, c = x.shape
        q, k, v = self.qkv(rms(x, self.norm1)).view(n, s, 3, self.heads, -1).unbind(2)
        rot = lambda t: torch.stack([-t[..., 1::2], t[..., 0::2]], -1).flatten(-2)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + self.out(a.transpose(1, 2).reshape(n, s, c))
        h, g = self.ffn1(rms(x, self.norm2)).chunk(2, -1)
        return x + self.ffn2(F.silu(h) * g)


class KataNet(nn.Module):
    def __init__(self, blocks=12, c=384, heads=6, ffn=1024, policy=48, value=96, hidden=128):
        super().__init__()
        self.spatial = nn.Conv2d(22, c, 3, padding=1, bias=False)
        self.glob = nn.Linear(19, c, bias=False)
        self.blocks = nn.ModuleList([Block(c, heads, ffn) for _ in range(blocks)])
        self.register_buffer("cos", torch.zeros(49, 1, c // heads))
        self.register_buffer("sin", torch.zeros(49, 1, c // heads))
        self.final = nn.Parameter(torch.zeros(4, c))
        self.p1, self.g1 = nn.Conv2d(c, policy, 1, bias=False), nn.Conv2d(c, policy, 1)
        self.pg, self.ppass = nn.Linear(3 * policy, policy, bias=False), nn.Linear(3 * policy, 4, bias=False)
        self.p2b, self.p2 = nn.Parameter(torch.zeros(policy)), nn.Conv2d(policy, 4, 1, bias=False)
        self.v1 = nn.Conv2d(c, value, 1)
        self.v2 = nn.Linear(3 * value, hidden)
        self.value, self.misc, self.more = nn.Linear(hidden, 3), nn.Linear(hidden, 10), nn.Linear(hidden, 8)
        self.own = nn.Conv2d(value, 1, 1, bias=False)

    def forward(self, spatial, glob):
        n = len(spatial)
        x = (self.spatial(spatial) + self.glob(glob)[:, :, None, None]).flatten(2).transpose(1, 2)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        mean, std, gamma, beta = self.final
        t = F.silu((x - mean) / std * gamma + beta).transpose(1, 2).reshape(n, -1, 7, 7)
        g = gpool(F.silu(self.g1(t)), False)
        p = F.silu(self.p1(t) + self.pg(g)[:, :, None, None] + self.p2b[:, None, None])
        policy = torch.cat([self.p2(p).flatten(2), self.ppass(g)[:, :, None]], 2)
        v = F.silu(self.v1(t))
        h = F.silu(self.v2(gpool(v, True)))
        return policy, self.value(h), self.misc(h), self.more(h), self.own(v)


def load(path):
    m = onnx.load(path)
    w = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in m.graph.initializer}
    consts = {n.output[0]: torch.from_numpy(numpy_helper.to_array(n.attribute[0].t).copy()) for n in m.graph.node if n.op_type == "Constant" and n.attribute[0].name == "value"}
    weight = {n.name: next(w[i] for i in n.input if i in w) for n in m.graph.node if n.op_type in ("MatMul", "Gemm") and any(i in w for i in n.input)}
    tables = [v for k, v in consts.items() if k.startswith("/model/blocks.0/") and v.numel() > 49 and v.numel() == 49 * v.shape[-1]]
    c = w["model.conv_spatial.weight"].shape[0]
    blocks = sum(k.startswith("model.blocks.") and k.endswith("norm1.weight") for k in w)
    net = KataNet(blocks, c, c // (tables[0].numel() // 49), weight["/model/blocks.0/ffn_linear1/MatMul"].shape[1],
                  w["model.policy_head.conv1p.weight"].shape[0], w["model.value_head.conv1.weight"].shape[0], w["model.value_head.linear2.weight"].shape[0])
    sd = net.state_dict()
    sd["spatial.weight"] = w["model.conv_spatial.weight"]
    sd["glob.weight"] = weight["/model/linear_global/MatMul"].T
    for i in range(blocks):
        b = f"/model/blocks.{i}/"
        sd[f"blocks.{i}.norm1"], sd[f"blocks.{i}.norm2"] = w[f"model.blocks.{i}.norm1.weight"], w[f"model.blocks.{i}.norm2.weight"]
        sd[f"blocks.{i}.qkv.weight"] = torch.cat([weight[b + f"{k}_proj/MatMul"].T for k in "qkv"])
        sd[f"blocks.{i}.out.weight"] = weight[b + "out_proj/MatMul"].T
        sd[f"blocks.{i}.ffn1.weight"] = torch.cat([weight[b + "ffn_linear1/MatMul"].T, weight[b + "ffn_linear_gate/MatMul"].T])
        sd[f"blocks.{i}.ffn2.weight"] = weight[b + "ffn_linear2/MatMul"].T
    sd["cos"], sd["sin"] = tables[0].view(49, 1, -1), tables[1].view(49, 1, -1)
    final = {n.name: next(w[i] for i in n.input if i in w) for n in m.graph.node if n.name.startswith("/model/norm_trunkfinal/")}
    sd["final"] = torch.stack([final[f"/model/norm_trunkfinal/{k}"].flatten() for k in ("Sub", "Div", "Mul", "Add")])
    sd["p1.weight"], sd["g1.weight"] = w["model.policy_head.conv1p.weight"], w["model.policy_head.conv1g.weight"]
    sd["g1.bias"] = w["model.policy_head.biasg.beta"].flatten()
    sd["pg.weight"], sd["ppass.weight"] = weight["/model/policy_head/linear_g/MatMul"].T, weight["/model/policy_head/linear_pass/MatMul"].T
    sd["p2b"], sd["p2.weight"] = w["model.policy_head.bias2.beta"].flatten(), w["model.policy_head.conv2p.weight"]
    sd["v1.weight"], sd["v1.bias"] = w["model.value_head.conv1.weight"], w["model.value_head.bias1.beta"].flatten()
    for a, b in (("v2", "linear2"), ("value", "linear_valuehead"), ("misc", "linear_miscvaluehead"), ("more", "linear_moremiscvaluehead")):
        sd[f"{a}.weight"], sd[f"{a}.bias"] = w[f"model.value_head.{b}.weight"], w[f"model.value_head.{b}.bias"]
    sd["own.weight"] = w["model.value_head.conv_ownership.weight"]
    net.load_state_dict(sd)
    return net.eval()


class Teacher:
    def __init__(self, path, gpus, floor=0.01, batch=4096):
        self.device, self.floor, self.batch = "cuda:0", floor, batch
        xy = torch.arange(49)
        dx, dy = xy[None] % 7 - xy[:, None] % 7, xy[None] // 7 - xy[:, None] // 7
        d = torch.maximum(dx.abs(), dy.abs())
        self.near, self.far = (d == 1).float().to(self.device), (d == 2).to(self.device)
        kind = torch.zeros(5, 5, dtype=torch.long)
        for k in range(16):
            kind[DX[k] + 2, DY[k] + 2] = k + 1
        self.move = (kind[(dx + 2).clamp(0, 4), (dy + 2).clamp(0, 4)] * 49 + xy[None]).to(self.device)
        self.models = [(torch.compile(load(path).half().to(f"cuda:{g}")), f"cuda:{g}") for g in gpus]
        self.pool = ThreadPoolExecutor(len(self.models))

    def forward(self, k, x, g):
        model, dev = self.models[k % len(self.models)]
        n = len(x)
        padding = -n % min(1024, 1 << (n - 1).bit_length())
        x, g = F.pad(x, (0, 0, 0, 0, 0, 0, 0, padding)), F.pad(g, (0, 0, 0, padding))
        with torch.no_grad():
            policy, value = model(x.to(dev).half(), g.to(dev).half())[:2]
        return policy[:n, 0, :49].float().to(self.device), value[:n].float().to(self.device)

    def run(self, own, opp, chosen=None):
        n = len(own)
        x = torch.zeros(n, 22, 49, device=self.device)
        g = torch.zeros(n, 19, device=self.device)
        x[:, 0], x[:, 1], x[:, 2] = 1, own, opp
        g[:, 3] = g[:, 9] = 1
        if chosen is not None:
            x[torch.arange(n, device=self.device), 4, chosen] = 1
            g[:, 0] = 1
        x = x.view(-1, 22, 7, 7)
        step = min(self.batch, -(-n // len(self.models)))
        parts = [(k, x[i:i + step], g[i:i + step]) for k, i in enumerate(range(0, n, step))]
        out = list(self.pool.map(lambda a: self.forward(*a), parts))
        return torch.cat([p for p, v in out]), torch.cat([v for p, v in out])

    def __call__(self, bb, side):
        bits = (bb[:, :, None] >> SHIFTS.to(self.device) & 1).bool()
        s = side.view(-1, 1).bool()
        own, opp = torch.where(s, bits[:, 1], bits[:, 0]), torch.where(s, bits[:, 0], bits[:, 1])
        empty = ~(own | opp)
        clone = empty & (own.float() @ self.near > 0)
        origin = own & (empty.float() @ self.far.float() > 0)
        logits, value = self.run(own.float(), opp.float())
        first = logits.masked_fill(~(clone | origin), -1e9).softmax(1)
        joint = torch.zeros(len(own), NMOVES, device=self.device)
        joint[:, :49] = first * clone
        row, sq = ((first > self.floor) & origin).nonzero(as_tuple=True)
        if len(row):
            reach = empty[row] & self.far[sq]
            second = self.run(own[row].float(), opp[row].float(), sq)[0].masked_fill(~reach, -1e9).softmax(1) * reach
            joint.index_put_((row[:, None].expand(-1, 49), self.move[sq]), first[row, sq, None] * second, accumulate=True)
        v = value.softmax(1)
        return joint / joint.sum(1, keepdim=True), v[:, 0] - v[:, 1]


def label(teacher, src, dst, gpus, chunk=16384):
    teacher = Teacher(teacher, range(gpus))
    dst.mkdir(parents=True, exist_ok=True)
    while True:
        for path in sorted(p for p in src.glob("*.npy") if ".tmp" not in p.name):
            rows = np.load(path)
            for i in range(0, len(rows), chunk):
                part = rows[i:i + chunk]
                policy, q = teacher(torch.from_numpy(part["bb"].astype(np.int64)).cuda(), torch.from_numpy(part["side"].astype(np.int64)).cuda())
                top = policy.topk(32, 1)
                cnt = (top.values * 65535).round()
                part["idx"], part["cnt"], part["n"] = top.indices.cpu().numpy(), cnt.cpu().numpy(), (cnt > 0).sum(1).cpu().numpy()
                q = (q * 32767).round().clamp(-32767, 32767)
                part["q"] = torch.where(q == 0, 1, q).cpu().numpy()
            np.save(dst / f"{path.stem}.tmp.npy", rows)
            os.rename(dst / f"{path.stem}.tmp.npy", dst / path.name)
            path.unlink()
        time.sleep(30)
